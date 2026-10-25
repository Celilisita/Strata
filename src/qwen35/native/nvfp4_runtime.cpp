#include "gdn_output_elision_scope.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include "strata/qwen35/native/weights.hpp"
#include "tiny_nvfp4.hpp"
#include "tiny32.hpp"
#include "tiny32_policy.hpp"
#include "warp_logical.hpp"
#include "warp_logical_policy.hpp"
#include "tiny_mma.hpp"
#include "tiny_mma_policy.hpp"
#include "fused_nvfp4.hpp"
#include "swiglu_vector.hpp"
#include "swiglu_vector_policy.hpp"
#include "half_output_gemm.hpp"
#include "resident_half_request.hpp"
#include "fp4_compensated_service.hpp"
#include "fp4_unpack_vector.hpp"
#include "half_tactic_policy.hpp"
#include "fp4_gu_fused.hpp"
#include "fp4_gu_tail_policy.hpp"
#include "fp4_gu_tma.hpp"
#include "halfmerge_policy.hpp"
#include "long_full_fusion_policy.hpp"
#include "half_tail_policy.hpp"
#include "fp4_tiny_fused.hpp"
#include "fp4_gu8_policy.hpp"
#include "fp4_residual_norm.hpp"
#include "fp4_residual_norm_scope.hpp"
#include "fp4_post_norm_store_policy.hpp"
#include "inputnorm_scope.hpp"
#include "qkvz_request_layout.hpp"
#include "ffn_down_add.hpp"
#include "nvfp4_tiny_head.hpp"
#include "q8_pp_sgemm.hpp"
#include "q36_startup_body_plan.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <cstdlib>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <cstdint>
namespace strata::qwen35::native {
static_assert(sizeof(Matrix)==24&&offsetof(Matrix,quant)==8&&offsetof(Matrix,rows)==12&&
    offsetof(Matrix,columns)==16&&offsetof(Matrix,layout)==20,"retained Q4/Q6 Matrix ABI");
namespace {
class Runtime final:public DenseGemm {
    DenseGemm &fallback_;cudaStream_t stream_;DeviceStorage storage_;
    Nvfp4Input input_{};Nvfp4Gemm gemm_;
    std::unique_ptr<Q8PpSgemm> q8_pp_;
    bool tiny_=false,fused_=false,warp_logical_=false;std::unique_ptr<Nvfp4HalfGemm> half_;
    Nvfp4Input gu_input_{};bool gu_available_=false,gu_active_=false,gu_tail_active_=false,gu_long_half_=false;
    GuFusionCounters gu_counters_;bool gu_tma_active_=false;GuTmaCounters gu_tma_counters_;
    bool gu_merge_active_=false;GuMergeCounters gu_merge_counters_;
    bool half_active_=false;unsigned long long half_submissions_=0;
    struct InventoryEntry {Nvfp4View view;bool packed,declared_packed;unsigned role;std::unique_ptr<GuTmaPlan> gu_tma_plan;std::unique_ptr<GuMergePlan> gu_merge_plan;};
    std::unordered_map<const void *,InventoryEntry> inventory_;
    unsigned long long calls_=0;
    bool vector_unpack_=false;uint64_t vector_unpack_calls_=0;
    bool residual_norm_active_=false,skip_post_norm_store_=false;ResidualNormCounters residual_norm_counters_;
    DownAddPolicy down_add_;
    TinyHeadCapability head_;
    bool tiny_fused_active_=false,gu8_active_=false;
    QkvzRequestLayout request_layout_;bool packed_mode_=false;void *packed_scratch_=nullptr;Nvfp4Input residual_{};CompensatedFfnCounters packed_counters_;
    CompensatedFp4Input compensated_input()const{return {input_,residual_};}
    bool packed(Matrix w)const{return inventory_.at(w.data).packed;}
    Nvfp4View canonical(Nvfp4View w,bool materialize,cudaStream_t stream){return qkvz_canonical_view(w,materialize,packed_scratch_,packed_counters_.scratch_bytes,head_.contains(w),vector_unpack_,vector_unpack_calls_,packed_counters_,stream);}
    const GuTmaPlan &tma_plan(const Nvfp4View &g,const Nvfp4View &u,cudaStream_t stream){
        auto &entry=inventory_.at(g.values);
        if(!entry.gu_tma_plan){
            cudaStreamCaptureStatus capture{};check_cuda(cudaStreamIsCapturing(stream,&capture),"GU cold descriptor admission");
            if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("GU TMA descriptor must precede Graph capture");
            // One descriptor lease in the authoritative gate-weight entry.
            // The first eager execution prepares it; Graph stores the maps by value.
            entry.gu_tma_plan=std::make_unique<GuTmaPlan>(make_gu_tma_plan({g},{u},gu_input_));
            ++gu_tma_counters_.plans;
        }
        return *entry.gu_tma_plan;
    }
    const GuMergePlan &merge_plan(const Nvfp4View &g,const Nvfp4View &u,cudaStream_t stream){
        auto &entry=inventory_.at(g.values);
        if(!entry.gu_merge_plan){
            cudaStreamCaptureStatus capture{};check_cuda(cudaStreamIsCapturing(stream,&capture),"merged GU cold descriptor admission");
            if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("merged GU descriptor must precede Graph capture");
            entry.gu_merge_plan=std::make_unique<GuMergePlan>(make_gu_merge_plan({g},{u},gu_input_));
            ++gu_merge_counters_.plans;
        }
        return *entry.gu_merge_plan;
    }
    bool prepare_norm(const Nvfp4View &g,const Nvfp4View &u,const Nvfp4View &d,const float *input,
                      int tokens,cudaStream_t stream,bool diagnostic,bool physical){
        auto *scope=residual_norm_pending();if(!scope)return false;
        scope->validate(input,g.columns,tokens,stream);bool encoded=false;
        if(residual_norm_active_&&!diagnostic&&physical&&g.rows==17408&&g.columns==5120&&
           u.rows==g.rows&&u.columns==g.columns&&g.input_scale==u.input_scale&&d.rows==5120&&d.columns==17408){
            // This admitted packed return consumes encoded input; observers keep the FP32 view.
            fp4_residual_norm(scope->a,scope->b,scope->gamma,scope->residual,fp4_post_norm_store_output(skip_post_norm_store_,gu_tma_active_,gu_merge_active_,tokens,scope->normalized),
                gu_input_,tokens,scope->epsilon,g.input_scale,stream);encoded=true;++residual_norm_counters_.fused;
        }else{
            if(scope->reference_fused)add_rms_norm(scope->a,scope->b,scope->gamma,scope->residual,scope->normalized,g.columns,tokens,scope->epsilon,stream);
            else{add(scope->a,scope->b,scope->residual,size_t(tokens)*g.columns,stream);
                rms_norm(scope->residual,scope->gamma,scope->normalized,g.columns,tokens,scope->epsilon,stream);}
            ++residual_norm_counters_.fallback;
        }
        scope->consume();return encoded;
    }
    const Nvfp4View &view(Matrix weight)const{
        request_layout_.validate();auto found=inventory_.find(weight.data);
        if(found==inventory_.end()||weight.quant!=Quant::NVFP4||
           weight.layout!=(found->second.declared_packed?BlockLayout::Nvfp4WarpFragments:BlockLayout::Nvfp4Planes)||
           weight.rows!=found->second.view.rows||weight.columns!=found->second.view.columns)
            throw std::invalid_argument("NVFP4 authoritative inventory/geometry mismatch");
        return found->second.view;
    }
public:
    bool output_elision_capability(Matrix w,Nvfp4View &result,Nvfp4Input*encoded)const{
        const auto it=inventory_.find(w.data);if(it==inventory_.end()||w.quant!=Quant::NVFP4||w.rows!=5120||w.columns!=6144)return false;
        result=it->second.view;if(encoded)*encoded=input_;return true;
    }

    Runtime(DenseGemm &fallback,const std::vector<Nvfp4View> &weights,int tokens,int columns,cudaStream_t stream,Matrix head)
        :fallback_(fallback),stream_(stream),gemm_(stream),head_(head){
        if(weights.empty()||tokens<16||columns%64)throw std::invalid_argument("NVFP4 runtime shape");
        if(head_.present()&&(columns<5120||tokens>8192))throw std::invalid_argument("tiny head runtime capacity");
        const char *fusion=std::getenv("STRATA_NVFP4_FUSED_ACT"),*half=std::getenv("STRATA_NVFP4_HALF_FFN");
        fused_=fusion&&std::string(fusion)=="1";
        if(fused_&&half&&std::string(half)=="1"){half_=std::make_unique<Nvfp4HalfGemm>(stream);half_active_=true;}
        const bool packed_ffn=std::any_of(weights.begin(),weights.end(),[](auto w){return compensated_ffn_published(w.values);});
        packed_mode_=packed_ffn||head_.present();
        if(packed_ffn){packed_counters_.scratch_bytes=size_t(17408)*5120/2;packed_scratch_=storage_.allocate(packed_counters_.scratch_bytes);}
        if(packed_mode_){
            packed_counters_.residual_bytes=nvfp4_value_bytes(16,columns)+nvfp4_scale_bytes(16,columns);
            residual_={storage_.allocate(nvfp4_value_bytes(16,columns)),storage_.allocate(nvfp4_scale_bytes(16,columns)),16,columns};
            check_cuda(cudaMemsetAsync(residual_.scales,0,nvfp4_scale_bytes(16,columns),stream),"compensated residual SF padding");
            // Resolve the new CUDA module before any request Graph capture.
            const auto pair=fp4_compensated_resources(true),linear=fp4_compensated_resources(false);
            const auto norm=fp4_residual_norm_resources();
            if(!norm.blocks||norm.local_bytes)throw std::runtime_error("residual norm resources");
            const auto fused=fp4_tiny_fused_resources();
            if(!fused.blocks||fused.local_bytes)throw std::runtime_error("Tiny fused codec resources");
            if(!pair.blocks||!linear.blocks||pair.local_bytes||linear.local_bytes)throw std::runtime_error("compensated provider resource admission");}
        input_={storage_.allocate(nvfp4_value_bytes(tokens,columns)),
            storage_.allocate(nvfp4_scale_bytes(tokens,columns)),tokens,columns};
        const char *tiny=std::getenv("STRATA_NVFP4_DIRECT_TINY");tiny_=tiny&&std::string(tiny)=="1";
        warp_logical_=warp_logical_supported();
        check_cuda(cudaMemsetAsync(input_.scales,0,nvfp4_scale_bytes(tokens,columns),stream),"NVFP4 input scale padding");
        auto *cold_output=static_cast<float *>(storage_.allocate(size_t(34816)*4));
        std::unordered_set<unsigned long long> warmed;
        for(auto weight:weights){
            const bool head=head_.admit(weight),physical=head||compensated_ffn_published(weight.values);
            if(physical&&!head&&!compensated_ffn_shape(weight)&&!compensated_projection_shape(weight)&&!qkvz_packed_shape(weight))throw std::invalid_argument("compensated physical geometry");
            if(!inventory_.emplace(weight.values,InventoryEntry{weight,physical,physical,qkvz_published_role(weight)}).second)throw std::invalid_argument("duplicate NVFP4 inventory");
            if(physical){++packed_counters_.matrices;qkvz_warm_cold(weight,warmed,gemm_,input_,cold_output,stream,[&](auto w){return canonical(w,true,stream);});
                gu_available_|=weight.rows==17408&&weight.columns==5120;continue;}
            // Cold plans for every single/tiny target shape; no allocation,
            // heuristics query or CUDA API setup during Graph capture.
            gemm_.prepare(weight,input_,1);
            const auto geometry=(static_cast<unsigned long long>(weight.rows)<<32)|unsigned(weight.columns);
            if(warmed.insert(geometry).second)gemm_.project(weight,input_,cold_output,1,stream);
        }
        head_.finish_admission();
        if(gu_available_){fp4_gu_merge_configure();const auto merged=fp4_gu_merge_resources(true);
            if(!merged.blocks||merged.local_bytes)throw std::runtime_error("merged GU resource admission");fp4_gu_tma_configure();const auto t=fp4_gu_tma_resources(true);
            if(!t.blocks||t.local_bytes)throw std::runtime_error("GU TMA resource admission");fp4_gu_fused_configure();const auto q=fp4_gu_fused_resources(false);
            if(!q.blocks||q.local_bytes)throw std::runtime_error("GU provider resource admission");}
        check_cuda(cudaStreamSynchronize(stream),"NVFP4 runtime allocation lease");
        storage_.release(cold_output);
    }
    ~Runtime()override{cudaStreamSynchronize(stream_);}
    void configure_q8_pp(const std::string &variant,const strata::qwen35::Spec &s,const DenseWeights &w,cudaStream_t stream){
        if(stream!=stream_||variant!="qwen36_27b"||q8_pp_)throw std::invalid_argument("Q8 PP requires its single Q36 Runtime/stream");
        q8_pp_=std::make_unique<Q8PpSgemm>(s,w,input_.capacity_rows,stream_);
    }
    bool q8_pp_pair(Matrix a,Matrix b,const float *x,float *y,float *z,int t,cudaStream_t s){
        if(!q8_pp_)return false;
        q8_pp_->project(a,b,x,y,z,t,s);return true;
    }
    Q8PpSgemmStats q8_pp_stats()const{return q8_pp_?q8_pp_->stats():Q8PpSgemmStats{};}
    void plan_q36_body_8192(const strata::qwen35::Spec &spec,cudaStream_t stream){
        if(stream!=stream_)throw std::invalid_argument("Q36 startup body plan ordered stream");
        prepare_q36_body_8192_plans(gemm_,input_,inventory_,
            [this](Nvfp4View w){return head_.contains(w);},spec,stream_,fused_,gu_available_);
    }
    void configure_head(bool enabled){head_.configure(enabled);}
    TinyHeadCounters head_counters()const{return head_.counters();}
    void project_head(Matrix m,const float *x,float *y,int t,cudaStream_t s){
        if(s!=stream_||t<1||t>input_.capacity_rows)throw std::invalid_argument("head ordered stream/span");
        head_.execute(view(m),compensated_input(),x,y,t,s);++calls_;
    }
    uint64_t vector_unpack_submissions()const{return vector_unpack_calls_;}
    bool layout_change_required(int prompt)const{return request_layout_.change_required(inventory_,prompt);}
    void prepare_request_layout(int prompt,bool cleared,cudaStream_t stream){request_layout_.prepare(inventory_,packed_scratch_,packed_counters_.scratch_bytes,stream_,stream,prompt,cleared);}
    void configure_half_request(bool resident,int prompt){qkvz_reset_request();long_full_fusion_requested();
        const char *tiny_fused=std::getenv("STRATA_NVFP4_TINY_FUSED_CODEC");
        if(tiny_fused&&std::string(tiny_fused)!="0"&&std::string(tiny_fused)!="1")throw std::invalid_argument("Tiny fused codec mode");
        tiny_fused_active_=packed_mode_&&(!tiny_fused||tiny_fused[0]!='0')&&input_.capacity_rows>=384&&input_.capacity_columns>=17408;
        gu8_active_=fp4_gu8_prepare(gu_available_&&tiny_fused_active_);
        const char *vector_mode=std::getenv("STRATA_NVFP4_VECTOR_UNPACK");
        if(vector_mode&&std::string(vector_mode)!="0"&&std::string(vector_mode)!="1")throw std::invalid_argument("vector unpack mode");
        vector_unpack_=packed_mode_&&(!vector_mode||vector_mode[0]!='0');
        const char *mode=std::getenv("STRATA_NVFP4_HALF_FFN");
        if(mode&&std::string(mode)!="0"&&std::string(mode)!="1"&&std::string(mode)!="resident")
            throw std::invalid_argument("NVFP4 half FFN mode");
        const bool enabled=!mode||std::string(mode)!="0";
        const bool forced=mode&&std::string(mode)=="1";
        half_active_=fused_&&enabled&&(forced||resident)&&prompt>=4096;
        // PREPARE has released both target and MTP graphs. Keep the allocation
        // between requests; only this request's admission changes.
        if(half_active_&&!half_)half_=std::make_unique<Nvfp4HalfGemm>(stream_);
        if(half_)half_->use_measured_tactic(half_active_&&resident&&half_tactic_requested());
        const char *gu=std::getenv("STRATA_NVFP4_GU_FUSION");
        if(gu&&std::string(gu)!="0"&&std::string(gu)!="1")throw std::invalid_argument("GU fusion mode");
        gu_active_=gu_available_&&fused_&&!half_active_&&(!gu||gu[0]!='0')&&prompt>=8192&&input_.capacity_rows>=8192;
        const char *tma=std::getenv("STRATA_NVFP4_GU_TMA");
        if(tma&&std::string(tma)!="0"&&std::string(tma)!="1")throw std::invalid_argument("GU TMA mode");
        gu_tma_active_=halfmerge_tma_prepare(gu_long_half_,gu_available_&&fused_&&half_active_&&resident,prompt,input_.capacity_rows,tma);
        const char *merge=std::getenv("STRATA_NVFP4_GU_MERGED");
        if(merge&&std::string(merge)!="0"&&std::string(merge)!="1")throw std::invalid_argument("merged GU mode");
        gu_merge_active_=gu_tma_active_&&(!merge||merge[0]!='0');
        gu_counters_.budget_fallback=false;
        if((gu_active_||gu_tma_active_)&&!gu_input_.values){
            const size_t values=nvfp4_value_bytes(8192,5120),scales=nvfp4_scale_bytes(8192,5120),bytes=values+scales;
            size_t free=0,total=0;check_cuda(cudaMemGetInfo(&free,&total),"GU PREPARE budget");
            if(bytes>free||free-bytes<DeviceStorage::reserve_bytes()){
                gu_active_=gu_tma_active_=gu_merge_active_=false;gu_counters_.budget_fallback=true;
            }else{
                auto *blob=static_cast<unsigned char*>(storage_.allocate(bytes));
                gu_input_={blob,blob+values,8192,5120};gu_counters_.input_bytes=bytes;
                check_cuda(cudaMemsetAsync(gu_input_.scales,0,scales,stream_),"GU input SF padding");
            }
        }
        gu_tail_active_=fp4_gu_tail_prepare(gu_active_);
        const char *norm=std::getenv("STRATA_NVFP4_RESIDUAL_NORM");
        if(norm&&std::string(norm)!="0"&&std::string(norm)!="1")throw std::invalid_argument("residual norm mode");
        const char *f32_norm=std::getenv("STRATA_NVFP4_F32_RESIDUAL_NORM");
        if(f32_norm&&std::string(f32_norm)!="0"&&std::string(f32_norm)!="1"&&std::string(f32_norm)!="auto")
            throw std::invalid_argument("F32 residual norm mode");
        residual_norm_active_=((gu_tma_active_&&gu_merge_active_&&(!gu_long_half_||long_full_fusion_requested()))||
            (gu_active_&&(!f32_norm||f32_norm[0]!='0')))&&(!norm||norm[0]!='0');
        skip_post_norm_store_=fp4_post_norm_store_prepare(residual_norm_active_&&gu_tma_active_&&gu_merge_active_);
        half_tail_prepare_request(resident&&gu_long_half_&&gu_tma_active_&&gu_merge_active_&&gu_input_.values,prompt);
    }
    ResidualNormCounters residual_norm_counters()const{return residual_norm_counters_;}
    void configure_down_add_request(bool enabled){down_add_.configure(enabled);}
    DownAddCounters down_add_counters()const{return down_add_.counters();}
    GuMergeCounters merge_counters()const{return gu_merge_counters_;}
    GuTmaCounters tma_counters()const{return gu_tma_counters_;}
    GuFusionCounters gu_counters()const{return gu_counters_;}
    unsigned long long half_submissions()const{return half_submissions_;}
    uint64_t half_tactic_submissions()const{return half_?half_->alternate_submissions():0;}
    CompensatedFfnCounters packed_counters()const{return packed_counters_;}
    bool direct(const Nvfp4View &w,int tokens)const{
        return tiny_&&tokens>=1&&tokens<=4&&w.columns%256==0&&
            size_t(tokens)*w.columns/32*NvTinyActivationBytes<=nvfp4_value_bytes(input_.capacity_rows,input_.capacity_columns);
    }
    void quantized(Matrix weight,const float *input,float *output,int tokens,cudaStream_t stream){
        const auto &w=view(weight);
        if(stream!=stream_||!input||!output)throw std::invalid_argument("NVFP4 ordered stream/operand lease");
        if(head_.contains(w)){project_head(weight,input,output,tokens,stream);return;}
        const bool normalized=gdn_output_elision_prepare(weight,w,input,output,input_,tokens,stream);
        if(packed(weight)&&!qkvz_tiny_fallback(w,tokens)){
            if(tokens<=3){
                fp4_compensated_quantize(input,compensated_input(),tokens,w.columns,w.input_scale,stream);qkvz_record_codec(w);
                qkvz_record_projection(w);fp4_compensated_linear({w},compensated_input(),output,tokens,stream);++packed_counters_.tiny_submissions;
            }else{
                if(!normalized)nvfp4_quantize(input,input_,tokens,w.columns,w.input_scale,stream);
                gemm_.project(canonical(w,true,stream),input_,output,tokens,stream);
            }
            ++calls_;return;
        }
        if(direct(w,tokens)){
            nvfp4_tiny_quantize(input,input_.values,w.columns,tokens,stream);
            (warp_logical_&&warp_logical_linear_enabled(tokens,w.columns,w.rows)?nvfp4_warp_logical_linear:
                (tiny_mma_linear_enabled(tokens,w.columns,w.rows)?nvfp4_tiny_mma_linear:
                (tiny32_linear_enabled(tokens,w.columns)?nvfp4_tiny32_linear:nvfp4_tiny_linear)))(canonical(w,packed(weight),stream),input_.values,output,tokens,stream);++calls_;return;
        }
        if(!normalized)nvfp4_quantize(input,input_,tokens,w.columns,w.input_scale,stream);
        gemm_.project(canonical(w,packed(weight),stream),input_,output,tokens,stream);++calls_;
    }
    void projection_set(const float* input,std::initializer_list<Nvfp4Projection> projections,int tokens,cudaStream_t stream){
        if(stream!=stream_||!input||projections.size()<1||projections.size()>4||tokens<1||tokens>input_.capacity_rows)
            throw std::invalid_argument("NVFP4 shared projection stream/operand lease");
        const int columns=projections.begin()->weight.columns;
        std::array<const Nvfp4View*,4> views{};size_t index=0;
        auto overlap=[](const void* a,size_t n,const void* b,size_t m){
            const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+m&&y<x+n;
        };
        for(const auto& p:projections){
            if(head_.contains(p.weight))throw std::invalid_argument("tiny-only head requires its projection lease");
            const auto& w=view(p.weight);views[index++]=&w;
            if(!p.output||w.columns!=columns||overlap(input,size_t(tokens)*columns*4,p.output,size_t(tokens)*w.rows*4))
                throw std::invalid_argument("NVFP4 shared projection geometry/output ownership");
            for(const auto& q:projections)if(&p!=&q&&overlap(p.output,size_t(tokens)*w.rows*4,q.output,size_t(tokens)*q.weight.rows*4))
                throw std::invalid_argument("NVFP4 shared projection output aliases");
        }
        // One ordered codec lease; mixed representations/scales encode again.
        if(packed(projections.begin()->weight))qkvz_record_group(views,projections,tokens);bool prepared=inputnorm_prepare_set(input,views,projections,input_,tokens,stream);int previous_kind=prepared?0:-1;float previous_scale=prepared?views[0]->input_scale:0;index=0;
        for(const auto &p:projections){
            const auto& w=*views[index++];const bool physical=packed(p.weight);
            const bool compensated=physical&&tokens<=3&&!qkvz_tiny_fallback(w,tokens),tiny=(!physical||qkvz_tiny_fallback(w,tokens))&&direct(w,tokens);
            const int kind=compensated?2:(tiny?1:0);
            if(!prepared||kind!=previous_kind||(kind!=1&&w.input_scale!=previous_scale)){
                if(compensated){fp4_compensated_quantize(input,compensated_input(),tokens,columns,w.input_scale,stream);qkvz_record_codec(w);}
                else if(tiny)nvfp4_tiny_quantize(input,input_.values,columns,tokens,stream);
                else nvfp4_quantize(input,input_,tokens,columns,w.input_scale,stream);
            }
            if(compensated){qkvz_record_projection(w);fp4_compensated_linear({w},compensated_input(),p.output,tokens,stream);++packed_counters_.tiny_submissions;}
            else if(tiny)(warp_logical_&&warp_logical_linear_enabled(tokens,w.columns,w.rows)?nvfp4_warp_logical_linear:
                (tiny_mma_linear_enabled(tokens,w.columns,w.rows)?nvfp4_tiny_mma_linear:
                (tiny32_linear_enabled(tokens,w.columns)?nvfp4_tiny32_linear:nvfp4_tiny_linear)))(canonical(w,physical,stream),input_.values,p.output,tokens,stream);
            else gemm_.project(canonical(w,physical,stream),input_,p.output,tokens,stream);
            ++calls_;prepared=true;previous_kind=kind;previous_scale=w.input_scale;
            if(p.completion)check_cuda(cudaEventRecord(p.completion,stream),"NVFP4 shared projection diagnostic completion");
        }
    }
    bool ffn(Matrix gate,Matrix up,Matrix down,const float *input,const Buffers &buffers,int tokens,cudaStream_t stream,bool diagnostic){
        if(head_.contains(gate)||head_.contains(up)||head_.contains(down))throw std::invalid_argument("tiny-only head is not an FFN operand");
        const auto &g=view(gate),&u=view(up),&d=view(down);
        if(stream!=stream_||g.columns!=u.columns||g.rows!=u.rows)
            throw std::invalid_argument("NVFP4 paired projection lease");
        const bool pg=packed(gate),pu=packed(up),pd=packed(down);
        if((pg||pu||pd)&&(!pg||!pu||!pd||g.input_scale!=u.input_scale))throw std::logic_error("packed FFN inventory agreement");
        auto *down_scope=down_add_.admit(buffers.output,g.columns,tokens,stream,
            (!gu_long_half_||long_full_fusion_requested())&&(gu_tma_active_?gu_merge_active_:gu_active_)&&pg&&pu&&pd&&tokens==8192&&!diagnostic&&
            g.rows==17408&&g.columns==5120&&d.rows==5120&&d.columns==17408);
        const bool normalized=prepare_norm(g,u,d,input,tokens,stream,diagnostic,pg&&pu&&pd);
        if(pg||pu||pd){
            if(tokens<=3){
                fp4_compensated_quantize(input,compensated_input(),tokens,g.columns,g.input_scale,stream);
                if(tiny_fused_active_&&!diagnostic&&(tokens==1||tokens==3)&&g.rows==17408&&g.columns==5120&&d.rows==5120&&d.columns==17408){
                    const auto out=fp4_tiny_fused_output(compensated_input(),g.rows);
                    fp4_gu8_project(gu8_active_,{g},{u},compensated_input(),d.input_scale,tokens,stream);
                    fp4_compensated_linear({d},out,buffers.output,tokens,stream);
                    ++tiny_fused_submissions()[tokens-1];calls_+=3;packed_counters_.tiny_submissions+=3;return false;
                }
                fp4_compensated_pair({g},{u},compensated_input(),buffers.gate,buffers.up,buffers.activation,tokens,stream);
                fp4_compensated_quantize(buffers.activation,compensated_input(),tokens,d.columns,d.input_scale,stream);
                fp4_compensated_linear({d},compensated_input(),buffers.output,tokens,stream);
                calls_+=3;packed_counters_.tiny_submissions+=3;return true;
            }
        }
        // The GU source and Down destination use distinct encoded leases:
        // CTAs must not overwrite the input that another CTA is still reading.
        // Preserve the faster measured Half path and the diagnostic materialization.
        if(gu_tma_active_&&pg&&pu&&pd&&(tokens==8192||half_tail_admit(gu_merge_active_,tokens,diagnostic))&&!diagnostic&&
           g.rows==17408&&g.columns==5120&&d.rows==5120&&d.columns==17408){
            const GuMergePlan *merged=gu_merge_active_?&merge_plan(g,u,stream):nullptr;
            const GuTmaPlan *base=gu_merge_active_?nullptr:&tma_plan(g,u,stream);
            if(!normalized)nvfp4_quantize(input,gu_input_,tokens,g.columns,g.input_scale,stream);
            if(merged){if(tokens==5424)half_tail_submit(*merged,{g},{u},gu_input_,input_,tokens,true,d.input_scale,stream);else halfmerge_dispatch(*merged,{g},{u},gu_input_,input_,tokens,true,d.input_scale,stream,gu_long_half_);++gu_merge_counters_.submissions;}
            else fp4_gu_tma(*base,{g},{u},gu_input_,input_,tokens,true,d.input_scale,stream);
            down_add_.project(gemm_,canonical(d,true,stream),input_,buffers.output,tokens,stream,down_scope);
            calls_+=3;++half_submissions_;++gu_tma_counters_.submissions;return false;
        }
        if(gu_active_&&pg&&pu&&pd&&(tokens==8192||(!half_active_&&gu_tail_active_&&tokens==5424))&&!diagnostic&&
           g.rows==17408&&g.columns==5120&&d.rows==5120&&d.columns==17408){
            if(!normalized)nvfp4_quantize(input,gu_input_,tokens,g.columns,g.input_scale,stream);
            fp4_gu_tail_project({g},{u},gu_input_,input_,tokens,false,d.input_scale,stream);
            down_add_.project(gemm_,canonical(d,true,stream),input_,buffers.output,tokens,stream,down_scope);
            calls_+=3;++gu_counters_.submissions;return false;
        }
        if(!pg&&direct(g,tokens)&&direct(u,tokens)){
            nvfp4_tiny_quantize(input,input_.values,g.columns,tokens,stream);
            (warp_logical_&&warp_logical_pair_enabled(tokens,g.columns,g.rows)?nvfp4_warp_logical_pair:
                (tiny_mma_pair_enabled(tokens,g.columns,g.rows)?nvfp4_tiny_mma_pair:
                (tiny32_pair_enabled(tokens,g.columns)?nvfp4_tiny32_pair:nvfp4_tiny_pair)))(g,u,input_.values,buffers.gate,buffers.up,buffers.activation,tokens,stream);calls_+=2;
            quantized(down,buffers.activation,buffers.output,tokens,stream);return true;
        }
        nvfp4_quantize(input,input_,tokens,g.columns,g.input_scale,stream);
        if(half_active_&&half_&&tokens>=4096){
            ++half_submissions_;
            if(diagnostic&&(!buffers.half_weights||!buffers.half_input||tokens>2*g.columns))
                throw std::invalid_argument("NVFP4 half diagnostic scratch capacity");
            void *gh=diagnostic?buffers.half_weights:buffers.gate,*uh=diagnostic?buffers.half_input:buffers.up;
            half_->project(canonical(g,pg,stream),input_,gh,tokens,stream);
            if(g.input_scale!=u.input_scale)nvfp4_quantize(input,input_,tokens,u.columns,u.input_scale,stream);
            half_->project(canonical(u,pu,stream),input_,uh,tokens,stream);calls_+=2;
            nvfp4_fused_swiglu_half(gh,uh,diagnostic?buffers.activation:nullptr,input_,tokens,g.rows,d.input_scale,stream);
            gemm_.project(canonical(d,pd,stream),input_,buffers.output,tokens,stream);++calls_;
            if(diagnostic){const size_t count=size_t(tokens)*g.rows;
                nvfp4_expand_half(gh,buffers.gate,count,stream);nvfp4_expand_half(uh,buffers.up,count,stream);}
            return diagnostic;
        }
        gemm_.project(canonical(g,pg,stream),input_,buffers.gate,tokens,stream);
        if(g.input_scale!=u.input_scale)nvfp4_quantize(input,input_,tokens,u.columns,u.input_scale,stream);
        gemm_.project(canonical(u,pu,stream),input_,buffers.up,tokens,stream);calls_+=2;
        if(fused_&&tokens>=128){
            const auto activation_kernel=use_vector_swiglu(tokens,g.rows)?nvfp4_swiglu_vector:nvfp4_fused_swiglu;
            activation_kernel(buffers.gate,buffers.up,diagnostic?buffers.activation:nullptr,input_,tokens,g.rows,d.input_scale,stream);
            gemm_.project(canonical(d,pd,stream),input_,buffers.output,tokens,stream);++calls_;return diagnostic;
        }
        swiglu(buffers.gate,buffers.up,buffers.activation,size_t(tokens)*g.rows,stream);
        quantized(down,buffers.activation,buffers.output,tokens,stream);
        return true;
    }
    void project(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t s)override{fallback_.project(w,x,y,r,k,t,s);}
    bool project_pair(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t s)override{return fallback_.project_pair(w,x,y,r,k,t,s);}
    bool can_fuse_swiglu(int r,int k,int t)const override{return fallback_.can_fuse_swiglu(r,k,t);}
    void project_swiglu(const void *w,const void *x,void *y,float *l,float *g,float *u,int r,int k,int t,cudaStream_t s)override{
        fallback_.project_swiglu(w,x,y,l,g,u,r,k,t,s);
    }
    bool can_fuse_quant_swiglu(Matrix g,Matrix u,int t)const override{return fallback_.can_fuse_quant_swiglu(g,u,t);}
    void project_quant_swiglu(Matrix g,Matrix u,const void *x,void *y,void *a,float *l,float *dg,float *du,int t,cudaStream_t s)override{
        fallback_.project_quant_swiglu(g,u,x,y,a,l,dg,du,t,s);
    }
    void batched_fp16(const void *a,const void *b,float *c,GemmBatch p,cudaStream_t s)override{fallback_.batched_fp16(a,b,c,p,s);}
    size_t workspace_bytes()const override{return storage_.bytes()+gemm_.bytes()+fallback_.workspace_bytes()+(half_?half_->bytes():0)+q8_pp_stats().resident_bytes;}
    unsigned long long calls()const override{return calls_+fallback_.calls();}
    GemmWorkspace borrowed_workspace()const override{return fallback_.borrowed_workspace();}
};
Runtime &runtime(DenseGemm &service){
    auto *p=dynamic_cast<Runtime *>(&service);
    if(!p)throw std::invalid_argument("NVFP4 model requires its configured runtime");return *p;
}
}
bool gdn_output_elision_capability(DenseGemm&s,Matrix w,Nvfp4View&view,Nvfp4Input*encoded){
    auto*p=dynamic_cast<Runtime*>(&s);return p&&p->output_elision_capability(w,view,encoded);
}
std::unique_ptr<DenseGemm> make_nvfp4_runtime(DenseGemm &fallback,const std::vector<Nvfp4View> &weights,
                                            int tokens,int columns,cudaStream_t stream){
    return make_nvfp4_runtime(fallback,weights,tokens,columns,stream,{});
}
std::unique_ptr<DenseGemm> make_nvfp4_runtime(DenseGemm &f,const std::vector<Nvfp4View> &w,int t,int k,cudaStream_t s,Matrix h){return std::make_unique<Runtime>(f,w,t,k,s,h);}
void nvfp4_configure_q8_pp(DenseGemm &service,const std::string &variant,const strata::qwen35::Spec &s,const DenseWeights &w,cudaStream_t stream){
    if(q8_pp_sgemm_requested())runtime(service).configure_q8_pp(variant,s,w,stream);
}
bool nvfp4_q8_pp_pair(DenseGemm &service,Matrix a,Matrix b,const float *x,float *y,float *z,int t,cudaStream_t stream){
    auto *p=dynamic_cast<Runtime *>(&service);return p&&p->q8_pp_pair(a,b,x,y,z,t,stream);
}
Q8PpSgemmStats nvfp4_q8_pp_stats(DenseGemm &service){auto *p=dynamic_cast<Runtime *>(&service);return p?p->q8_pp_stats():Q8PpSgemmStats{};}
void nvfp4_plan_q36_body_8192(DenseGemm &s,const strata::qwen35::Spec &spec,cudaStream_t stream){
    runtime(s).plan_q36_body_8192(spec,stream);
}
bool nvfp4_request_layout_change_required(DenseGemm &s,int prompt){auto *p=dynamic_cast<Runtime*>(&s);return p&&p->layout_change_required(prompt);}
void nvfp4_prepare_request_layout(DenseGemm &s,int prompt,bool cleared,cudaStream_t stream){if(auto *p=dynamic_cast<Runtime*>(&s))p->prepare_request_layout(prompt,cleared,stream);}
void nvfp4_project_head(DenseGemm &s,Matrix w,const float *x,float *y,int t,cudaStream_t stream){runtime(s).project_head(w,x,y,t,stream);}
void nvfp4_configure_head_request(DenseGemm &s,bool enabled){if(auto *p=dynamic_cast<Runtime*>(&s))p->configure_head(enabled);}
TinyHeadCounters nvfp4_head_counters(DenseGemm &s){auto *p=dynamic_cast<Runtime*>(&s);return p?p->head_counters():TinyHeadCounters{};}
void nvfp4_configure_half_request(DenseGemm &service,bool resident,int prompt){
    if(auto *p=dynamic_cast<Runtime *>(&service))p->configure_half_request(resident,prompt);
}
unsigned long long nvfp4_half_submissions(DenseGemm &service){
    auto *p=dynamic_cast<Runtime *>(&service);return p?p->half_submissions():0;
}
ResidualNormCounters nvfp4_residual_norm_counters(DenseGemm &service){
    auto *p=dynamic_cast<Runtime*>(&service);return p?p->residual_norm_counters():ResidualNormCounters{};
}
void nvfp4_configure_down_add_request(DenseGemm &service,bool enabled){
    if(auto *p=dynamic_cast<Runtime*>(&service))p->configure_down_add_request(enabled);
}
DownAddCounters nvfp4_down_add_counters(DenseGemm &service){
    auto *p=dynamic_cast<Runtime*>(&service);return p?p->down_add_counters():DownAddCounters{};
}
CompensatedFfnCounters nvfp4_compensated_counters(DenseGemm &service){
    auto *p=dynamic_cast<Runtime *>(&service);return p?p->packed_counters():CompensatedFfnCounters{};
}
uint64_t nvfp4_half_tactic_submissions(DenseGemm &service){
    auto *p=dynamic_cast<Runtime *>(&service);return p?p->half_tactic_submissions():0;
}
uint64_t nvfp4_vector_unpack_submissions(DenseGemm &service){
    auto *p=dynamic_cast<Runtime *>(&service);return p?p->vector_unpack_submissions():0;
}
GuMergeCounters nvfp4_gu_merge_counters(DenseGemm &service){
    auto *p=dynamic_cast<Runtime *>(&service);return p?p->merge_counters():GuMergeCounters{};
}
GuTmaCounters nvfp4_gu_tma_counters(DenseGemm &service){
    auto *p=dynamic_cast<Runtime *>(&service);return p?p->tma_counters():GuTmaCounters{};
}
GuFusionCounters nvfp4_gu_fusion_counters(DenseGemm &service){
    auto *p=dynamic_cast<Runtime *>(&service);return p?p->gu_counters():GuFusionCounters{};
}
bool nvfp4_shared_projections_enabled(){
    static const bool enabled=[](){const char* p=std::getenv("STRATA_NVFP4_SHARE_PROJECTIONS");return !p||p[0]!='0';}();
    return enabled;
}
void nvfp4_project_set(DenseGemm& service,const float* input,std::initializer_list<Nvfp4Projection> projections,int tokens,cudaStream_t stream){
    runtime(service).projection_set(input,projections,tokens,stream);
}
void nvfp4_project(DenseGemm &service,Matrix weight,const float *input,float *output,int tokens,cudaStream_t stream){
    runtime(service).quantized(weight,input,output,tokens,stream);
}
void nvfp4_ffn(DenseGemm &service,Matrix g,Matrix u,Matrix d,const float *input,const Buffers &buffers,int tokens,cudaStream_t stream){
    runtime(service).ffn(g,u,d,input,buffers,tokens,stream,true);
}
bool nvfp4_ffn_fused(DenseGemm &s,Matrix g,Matrix u,Matrix d,const float *x,const Buffers &b,int t,cudaStream_t stream,bool diagnostic){return runtime(s).ffn(g,u,d,x,b,t,stream,diagnostic);}
}
