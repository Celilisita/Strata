#include "strata/runtime/capacity_storage.hpp"
#pragma once
#include "strata/qwen35/native/ffn.hpp"
#include "strata/qwen35/native/weights.hpp"
#include "strata/gemma4/native/q6_down.hpp"
#include <cublas_v2.h>
#include <cstdint>
namespace strata::gemma4::native {
namespace primitive=strata::qwen35::native;
struct SingleQ8View;
enum class ShortPpBackend { ExistingPanel3=0, FastPanel3=1, FastFullK=2 };
enum class T1Q4FfnPolicy { Off=0, GateUp=1, Both=2 };
struct AttentionWorkspace {void* data;size_t bytes;cublasHandle_t handle;void* blas_data;size_t blas_bytes;};
struct FfnWeights {
    primitive::Matrix gate,up,down;
    const float *pre_norm,*post_norm,*layer_scale;
};
struct FfnBuffers {float *normalized,*gate,*up,*activation,*down,*post;};
// Initial exact-F32 GPU path. cuBLAS SGEMM preserves the admitted input
// precision while Dense Gemma's GELU/norm/residual state is brought up.
// This is a functional vertical slice, not a high-performance model claim.
class Ffn {
    strata::runtime::CapacityStorage storage_;
    FfnWeights weights_;
    FfnBuffers buffers_{};
    cublasHandle_t handle_=nullptr;
    float* decoded_=nullptr;
    void* workspace_=nullptr;
    int hidden_,intermediate_,maximum_;
    float epsilon_;
    bool single_ffn_=false,row_half_=false,fast_ffn_=false,compact_half_=false,three_panel_=false;
    bool residual_pp_=false,vector3_=false,all_pp_=false,full_down_=false,grouped_=false,paired_pp_=false,native_pp_=false;
    T1Q4FfnPolicy t1_q4_policy_=T1Q4FfnPolicy::Off;
    // Logical eligibleQ4GU dispatch count:244/269/291; leaf counters are separate.
    size_t t1_q4_gate_up_enqueues_=0,t1_q4_down_enqueues_=0;
    bool t1_gate_up_word_=false,t1_gate_up_direct_=false;
    size_t t1_gu291_enqueues_=0;
    std::uint64_t t1_gu291_epoch_=0;
    size_t t1_gu269_enqueues_=0;
    std::uint64_t t1_gu269_epoch_=0;
    bool t1_q6_down_striped_=false;
    size_t t1_q6_down_enqueues_=0;
    std::uint64_t t1_q6_down_epoch_=0;
    ShortPpBackend short_pp_backend_=ShortPpBackend::ExistingPanel3;
    size_t short_pp_enqueues_[3]{};
    std::uint64_t short_pp_epoch_=0;
    std::uint64_t prepared_epoch_=0;
    int residual_slice_=512,pp_algorithm_=-1;
    void project(primitive::Matrix,const float*,float*,int,cudaStream_t,bool=true,bool=false);
public:
    Ffn(FfnWeights,int maximum,float epsilon);
    ~Ffn();
    Ffn(const Ffn&)=delete;
    Ffn& operator=(const Ffn&)=delete;
    void rebind(FfnWeights);
    void select_t1_q4_ffn(T1Q4FfnPolicy policy){
        if(policy!=T1Q4FfnPolicy::Off&&policy!=T1Q4FfnPolicy::GateUp&&policy!=T1Q4FfnPolicy::Both)
            throw std::invalid_argument("Gemma T1 Q4 FFN policy enum");
        if(policy!=T1Q4FfnPolicy::Off&&(!single_ffn_||hidden_!=5376||intermediate_!=21504))
            throw std::invalid_argument("Gemma striped T1 Q4 requires fixed Single40 FFN");
        if(policy==T1Q4FfnPolicy::Off&&t1_gate_up_direct_)
            throw std::invalid_argument("Gemma291 requires eligible Q4 GU policy");
        t1_q4_policy_=policy;
    }
    T1Q4FfnPolicy t1_q4_ffn_policy()const{return t1_q4_policy_;}
    size_t t1_q4_gate_up_enqueues()const{return t1_q4_gate_up_enqueues_;}
    size_t t1_q4_down_enqueues()const{return t1_q4_down_enqueues_;}
    void clear_t1_q4_enqueues(){t1_q4_gate_up_enqueues_=0;t1_q4_down_enqueues_=0;}
    // Cold owner policy. Model enforces P0/Graph invalidation before calling.
    void select_t1_q6_down_striped(bool enabled){
        if(enabled&&(!single_ffn_||hidden_!=5376||intermediate_!=21504))
            throw std::invalid_argument("Gemma262 requires fixed Single40 FFN");
        t1_q6_down_striped_=enabled;
    }
    bool t1_q6_down_striped_enabled()const{return t1_q6_down_striped_;}
    size_t t1_q6_down_enqueues()const{return t1_q6_down_enqueues_;}
    void clear_t1_q6_down_enqueues(){t1_q6_down_enqueues_=0;}
    // Model enforces coldP0/Graph identity; only the existing244eligible branch uses it.
    void select_t1_gate_up_word(bool enabled){
        if(enabled&&t1_gate_up_direct_)throw std::invalid_argument("Gemma269/291 mutually exclusive GU");
        if(enabled&&(!single_ffn_||hidden_!=5376||intermediate_!=21504))
            throw std::invalid_argument("Gemma269 requires fixed Single40 FFN");
        t1_gate_up_word_=enabled;
    }
    bool t1_gate_up_word_enabled()const{return t1_gate_up_word_;}
    size_t t1_gu269_enqueues()const{return t1_gu269_enqueues_;}
    void clear_t1_gu269_enqueues(){t1_gu269_enqueues_=0;}
    // Cold owner policy only; Model validates P0 and releases all old Graphs.
    void select_t1_gate_up_direct(bool enabled){
        if(enabled){
            if(t1_gate_up_word_||!single_ffn_||t1_q4_policy_==T1Q4FfnPolicy::Off||hidden_!=5376||intermediate_!=21504||
                !decoded_||reinterpret_cast<std::uintptr_t>(decoded_)%8)
                throw std::invalid_argument("Gemma291 requires fixed eligible Single40 and exclusive GU");
            for(auto w:{weights_.gate,weights_.up})if(!w.data||w.quant!=primitive::Quant::Q4K||
                w.layout!=primitive::BlockLayout::Gguf||w.rows!=21504||w.columns!=5376||reinterpret_cast<std::uintptr_t>(w.data)%16)
                throw std::invalid_argument("Gemma291 raw Q4 weight16/input8 inventory");
        }
        t1_gate_up_direct_=enabled;
    }
    bool t1_gate_up_direct_enabled()const{return t1_gate_up_direct_;}
    bool single_ffn_enabled()const{return single_ffn_;}
    size_t t1_gu291_enqueues()const{return t1_gu291_enqueues_;}
    void clear_t1_gu291_enqueues(){t1_gu291_enqueues_=0;}
    void select_single_ffn(bool enabled){
        if(!enabled&&t1_gate_up_direct_)throw std::invalid_argument("Gemma291 requires Single40 FFN");
        if(!enabled&&t1_gate_up_word_)
            throw std::invalid_argument("Gemma269 requires Single40 FFN");
        if(!enabled&&t1_q6_down_striped_)
            throw std::invalid_argument("Gemma262 requires Single40 FFN");
        if(!enabled&&t1_q4_policy_!=T1Q4FfnPolicy::Off)
            throw std::invalid_argument("Gemma striped T1 Q4 requires Single40 FFN");
        if(!enabled&&is_aligned_q6_down(weights_.down))
            throw std::invalid_argument("Gemma native Q6 Down requires Single40 FFN");
        single_ffn_=enabled;
    }
    void select_fast_ffn(bool enabled){fast_ffn_=enabled;}
    void select_row_half(bool enabled){
        if(enabled&&(pp_algorithm_!=2||!residual_pp_))throw std::invalid_argument("Gemma row-half requires FP16 PP");
        row_half_=enabled;
    }
    void select_compact_half(bool enabled){
        if(enabled&&(pp_algorithm_!=2||!residual_pp_))
            throw std::invalid_argument("Gemma compact panels require FP16 PP");
        compact_half_=enabled;
    }
    // Cold owner policy; only the existing1280 row branch consumes it.
    void select_three_panel_half(bool enabled){
        if(enabled&&(pp_algorithm_!=2||!residual_pp_||hidden_!=5376||intermediate_!=21504||maximum_<1280))
            throw std::invalid_argument("Gemma three-panel requires FP16 PP and fixed5376/21504/1280 extent");
        three_panel_=enabled;
    }
    bool three_panel_half_enabled()const{return three_panel_;}
    // Cold Model-controlled policy. Runtime admission is enforced by the finite driver.
    void select_short_pp_backend(ShortPpBackend mode){
        if(mode!=ShortPpBackend::ExistingPanel3&&mode!=ShortPpBackend::FastPanel3&&mode!=ShortPpBackend::FastFullK)
            throw std::invalid_argument("Gemma305 short PP backend enum");
        if(mode!=ShortPpBackend::ExistingPanel3&&(pp_algorithm_!=2||!residual_pp_||
            hidden_!=5376||intermediate_!=21504||maximum_!=8192||!decoded_||!workspace_||!handle_))
            throw std::invalid_argument("Gemma305 fast short PP requires fixed RN16/PP3 geometry");
        short_pp_backend_=mode;
    }
    ShortPpBackend short_pp_backend()const{return short_pp_backend_;}
    size_t short_pp_enqueues(ShortPpBackend mode)const{
        const auto i=static_cast<unsigned>(mode);
        if(i>2)throw std::invalid_argument("Gemma305 short PP counter enum");return short_pp_enqueues_[i];
    }
    void clear_short_pp_enqueues(){for(auto& x:short_pp_enqueues_)x=0;}

    void select_periodic_row(bool enabled){
        if(enabled)throw std::invalid_argument("Gemma periodic row unavailable in projection-only build");
    }
    void row_half_prefill(const float*,int,cudaStream_t);
    // The view expires at the next preparation/workspace use on this Ffn.
    // Q/K/V consumers are enqueued together on the owner's ordered stream.
    SingleQ8View prepare_linear_single(const float*,int,int,cudaStream_t);
    void linear_single(primitive::Matrix,SingleQ8View,float*,cudaStream_t);
    void select_native_pp(bool enabled);
    void select_paired_pp(bool enabled){paired_pp_=enabled;}
    void pair_prefill(const float* normalized,int tokens,cudaStream_t);
    void select_half_pp(bool enabled){pp_algorithm_=enabled?2:-1;}
    void select_pp_algorithm(int id){if(id< -1||id>1)throw std::invalid_argument("Gemma library algorithm policy");pp_algorithm_=id;}
    void select_main_slice(int slice){if(slice!=0&&slice!=512&&slice!=896&&slice!=1024)throw std::invalid_argument("Gemma admitted PP slices");residual_slice_=slice;}
    void select_full_down(bool enabled){full_down_=enabled;}
    void select_all_pp(bool enabled){all_pp_=enabled;}
    void select_grouped(bool enabled){grouped_=enabled;}
    void select_vector3(bool enabled){vector3_=enabled;}
    bool prepare_vector_input(const float*,int,int,cudaStream_t);
    AttentionWorkspace attention_workspace()const{return {decoded_,size_t(hidden_)*intermediate_*4,handle_,workspace_,32u*1024*1024};}
    // Cold benchmark inspection only; never invoked by model execution.
    void inspect_residual_input(float* host_two,cudaStream_t)const;
    void select_residual_pp(bool enabled,int slice=512){residual_pp_=enabled;residual_slice_=slice;}
    void linear(primitive::Matrix m,const float* x,float* y,int tokens,cudaStream_t stream,bool eligible=true,bool prepared=false){project(m,x,y,tokens,stream,eligible,prepared);}
    void run(const float* after_attention,float* output,int tokens,cudaStream_t);
    const FfnBuffers& buffers()const{return buffers_;}
};
struct PairPrefillLease {
    const float* source=nullptr;
    float *gate=nullptr,*up=nullptr;
    void *weight_arena=nullptr,*input_arena=nullptr,*workspace=nullptr;
    size_t weight_bytes=0,input_bytes=0,output_stride=0;
    int tokens=0,slice=1024,algorithm=-1;
    cublasHandle_t handle=nullptr;cudaStream_t stream=nullptr;
    const void* owner=nullptr;std::uint64_t epoch=0;
};
void configure_native_pair();
void native_pair_project(primitive::Matrix,primitive::Matrix,const PairPrefillLease&);
void paired_library_project(primitive::Matrix,primitive::Matrix,const PairPrefillLease&);
void half_library_project(primitive::Matrix,const float*,float*,int,void*,void*,cublasHandle_t,void*,cudaStream_t,int);
// The exact production RN16 weight reader, also used by bounded GPU fixtures.
void prepare_half_library_weights(primitive::Matrix,void*,size_t,cudaStream_t);
void residual_library_project(primitive::Matrix,const float*,float*,int,void*,void*,cublasHandle_t,void*,cudaStream_t,int,int=-1);
void decode_float(primitive::Matrix,float*,cudaStream_t);
void gelu_product(const float*,const float*,float*,size_t,cudaStream_t);
void residual_scale(const float*,const float*,const float*,float*,size_t,cudaStream_t);
void softcap_logits(float*,size_t,float,cudaStream_t);
}
