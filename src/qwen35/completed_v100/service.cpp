#include "q35_shared_prepare.hpp"
#include "strata/qwen35/completed_v100/service.hpp"
#include "attention_online_policy.hpp"
#include "q6_head_policy.hpp"
#include "mtp_abort.hpp"
#include "pp_prefetch.hpp"
#include "strata/qwen35/completed_v100/native_options.hpp"
#include "strata/models/request_sampling.hpp"
#include "strata/qwen35/completed_v100/native/mtp.hpp"
#include "service_profile.hpp"
#include "gdn_subwarp8.hpp"
#include "pp_long_library_option.hpp"
#include "attention_verify_online.hpp"
#include "service_decode_graph.hpp"
#include "service_graph_cache.hpp"
#include "attention_verify_dispatch.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_gemm.hpp"
#include "strata/qwen35/completed_v100/native/attention_short32.hpp"
#include "strata/qwen35/completed_v100/native/gdn_subwarp16.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <cstdlib>

namespace {
using namespace strata::completed_v100::qwen35::native;
using Clock = std::chrono::steady_clock;
struct Stream {
    cudaStream_t p = nullptr;
    void open() { check_cuda(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking), "Dense service stream"); }
    ~Stream() { if (p) { cudaStreamSynchronize(p); cudaStreamDestroy(p); } }
};
std::vector<int> parse_ids(const std::string& csv, int vocab) {
    if (csv.empty() || csv.back() == ',') throw std::invalid_argument("empty token list");
    std::istringstream input(csv); std::string piece; std::vector<int> ids;
    while (std::getline(input, piece, ',')) {
        size_t used = 0; int id = std::stoi(piece, &used);
        if (used != piece.size() || id < 0 || id >= vocab) throw std::invalid_argument("token outside vocabulary");
        ids.push_back(id);
    }
    return ids;
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
class DenseService {
    const strata::completed_v100::models::NativeOptions options_;
    strata::GgufFile file_;
    strata::completed_v100::qwen35::Spec spec_;
    Stream stream_;
    std::unique_ptr<DenseWeights> weights_;
    std::unique_ptr<DenseGemm> fallback_, gemm_;
    std::unique_ptr<FfnShapePolicy> shape_;
    std::unique_ptr<DenseGemm> original_gemm_;
    std::unique_ptr<DenseModel> model_;
    DeviceStorage storage_;
    int* input_ = nullptr;
    int* history_ = nullptr;
    std::unique_ptr<strata::kernels::RadixSampler> sampler_;
    strata::models::RequestSampling sampling_;
    std::unique_ptr<NativeMtp> mtp_;
    std::unique_ptr<ServiceDecodeGraph> graph_;
    ServiceGraphCache graph_cache_;
    PpLongLibraryOption pp_library_;
    int position_ = -1, pending_ = -1, remaining_ = 0;
    bool active_ = false, speculative_ = false;
    int request_mtp_=0;
    bool wide_requested_=false,online_requested_=false,skip_pp_requested_=false,pp_tune_requested_=false,pp_s16_requested_=false,pp_dual_requested_=false,pp_s8_requested_=false;
    bool sampled_mtp_option_ = false;
    bool configured_library_=false,configured_packed_=false;
    bool reuse_eager_option_ = false, reused_prefill_ = false;
    bool reuse_eager_request_ = false;
    long long pp_eager_us_=0,pp_capture_us_=0,pp_replay_us_=0;
    bool q36_online_requested_=false,q36_online_selected_=false,service_abort_=false;
    MtpCounters request_counts_{};
    int pick() {
        int result = -1; unsigned invalid = 0;
        check_cuda(cudaMemcpyAsync(&invalid, model_->invalid_logits(), 4, cudaMemcpyDeviceToHost, stream_.p), "Dense finite flag");
        if (sampling_.enabled()) {
            if (!sampler_) sampler_ = std::make_unique<strata::kernels::RadixSampler>(int(spec_.n_vocab));
            const size_t bytes = sampling_.history.size() * sizeof(int);
            if (bytes) check_cuda(cudaMemcpyAsync(history_, sampling_.history.data(), bytes, cudaMemcpyHostToDevice, stream_.p), "Dense penalty history");
            sampler_->sample(model_->logits(), history_, int(sampling_.history.size()), sampling_.params, stream_.p);
            strata::kernels::SamplePick sample;
            check_cuda(cudaMemcpyAsync(&sample, sampler_->result(), sizeof(sample), cudaMemcpyDeviceToHost, stream_.p), "Dense sampled response");
            check_cuda(cudaStreamSynchronize(stream_.p), "Dense sampling completion");
            if (sample.first_bad != -1) throw std::runtime_error("Dense nonfinite logits");
            result = sample.id; ++sampling_.params.counter;
            // The device-picked state input, not just the HTTP response, changes.
            check_cuda(cudaMemcpyAsync(const_cast<int*>(model_->picked_token()), &result, 4,
                                       cudaMemcpyHostToDevice, stream_.p), "Dense sampled next-state input");
        } else check_cuda(cudaMemcpyAsync(&result, model_->picked_token(), 4, cudaMemcpyDeviceToHost, stream_.p), "Dense greedy response");
        check_cuda(cudaStreamSynchronize(stream_.p), "Dense token completion");
        if (invalid || result < 0 || result >= int(spec_.n_vocab)) throw std::runtime_error("Dense invalid logits or token");
        return result;
    }
    void release() {
        // Graphs borrow KV, prefix input and sampler state; drain them first.
        graph_.reset();
        if(q36_online_selected_||service_abort_||q6_head_warps()||
           (original_gemm_&&static_cast<Q35SharedNvfp4*>(original_gemm_.get())->shared_prepare()))
            graph_cache_.invalidate(mtp_.get());
        graph_cache_.release(mtp_.get());
        check_cuda(cudaStreamSynchronize(stream_.p), "Dense request drain");
        if(mtp_)mtp_abort::clear(*mtp_);
        service_abort_=q36_online_requested_=q36_online_selected_=false;set_q6_head_warps(0);
        model_->attention_work().use_verify_attention_kernel(service_verify);
        if(auto* shared=dynamic_cast<Q35SharedNvfp4*>(original_gemm_.get()))
            shared->shared_prepare(false,stream_.p);
        pp_prefetch_reset(original_gemm_.get(),stream_.p);
        model_->set_visual_input(nullptr,stream_.p);
        model_->reset_prefill(stream_.p);model_->select_q6_soa(false);
        pp_library_.reset();active_ = speculative_ = false;wide_requested_=false;online_requested_=false;skip_pp_requested_=false;pp_tune_requested_=false;pp_s16_requested_=false;pp_dual_requested_=false;pp_s8_requested_=false; position_ = pending_ = -1;
        remaining_ = 0; sampling_ = {};
        reuse_eager_request_=reuse_eager_option_;request_mtp_=options_.mtp;
        if(original_gemm_) {
            auto* provider=static_cast<Nvfp4Gemm*>(original_gemm_.get());
            provider->packed_decode(configured_packed_);provider->library_decode(configured_library_);
            provider->fused_cutlass(false,false);provider->q8_dp4a(0);provider->q8_warps(0);provider->q8_pp_tail(false);provider->q8_pp_long(false);provider->q8_pp_vector(false);provider->q8_pp_pair(false);provider->q8_pp_gdn(false);provider->q8_pp_lt(false);provider->q8_pp_fused(false);provider->q8_pp_decode8(false);provider->q8_pair_half(false);provider->q8_head_wide(false);provider->q8_pp_p2(false);provider->q8_fastsoa(false);provider->q8_soa(false);provider->q8_fast(false);provider->q8_pair(false);provider->q8_activation(0);provider->q8_optimized(false);provider->mixed_decode(false);provider->q8_blocked(false);
        }
    }
public:
    explicit DenseService(strata::completed_v100::models::NativeOptions o) : options_(std::move(o)), file_(options_.model) {
        if(const char* value=std::getenv("STRATA_DENSE_SAMPLED_MTP")) {
            if(std::string(value)!="0"&&std::string(value)!="1")throw std::invalid_argument("Invalid sampled MTP option");
            sampled_mtp_option_=std::string(value)=="1";
        }
        if(const char* value=std::getenv("STRATA_DENSE_REUSE_EAGER_PP")) {
            if(std::string(value)!="0"&&std::string(value)!="1")throw std::invalid_argument("Invalid eager PP reuse option");
            reuse_eager_option_=std::string(value)=="1";
        }
        reuse_eager_request_=reuse_eager_option_;request_mtp_=options_.mtp;
        std::string error;
        if (!strata::completed_v100::qwen35::load_spec(file_, spec_, error)) throw std::runtime_error(error);
        if (options_.context != int(spec_.context_length))
            throw std::invalid_argument("Dense native profile requires the model context length");
        stream_.open();
        weights_ = std::make_unique<DenseWeights>(file_, spec_, stream_.p);
        fallback_ = make_cublas_pair_gemm(stream_.p);
        gemm_ = make_cutlass_swiglu_gemm(*fallback_, stream_.p);
        shape_ = std::make_unique<FfnShapePolicy>(*gemm_, 8192);
        auto limits=service_limits(options_.context, options_.kv == "ram8k-reuse");
        if(weights_->recurrent[0].qkv.quant==Quant::NVFP4) {
            limits.native32=false;limits.joint_ffn=false;limits.joint_projections=false;limits.folded_ffn=false;
            original_gemm_=std::make_unique<Q35SharedNvfp4>(size_t(spec_.n_ff)*spec_.n_embd,int(spec_.n_ff),8192);
            if(const char* value=std::getenv("STRATA_NVFP4_PACKED_DECODE")) {
                const std::string flag(value);
                if(flag!="0"&&flag!="1")throw std::invalid_argument("Invalid packed NVFP4 GPU option");
                static_cast<Nvfp4Gemm*>(original_gemm_.get())->packed_decode(flag=="1");
            }
        }
        if(original_gemm_)if(const char* value=std::getenv("STRATA_NVFP4_LIBRARY_DECODE")) {
            const std::string flag(value);
            if(flag!="0"&&flag!="1")throw std::invalid_argument("Invalid NVIDIA library GPU option");
            static_cast<Nvfp4Gemm*>(original_gemm_.get())->library_decode(flag=="1");
        }
        if(original_gemm_) {
            const auto* provider=static_cast<Nvfp4Gemm*>(original_gemm_.get());
            configured_library_=provider->library_decode();configured_packed_=provider->packed_decode();
        }
        model_ = std::make_unique<DenseModel>(spec_, *weights_, limits, stream_.p);
        if(original_gemm_)model_->use_original_gemm(original_gemm_.get());
        input_ = static_cast<int*>(storage_.allocate(size_t(options_.context) * 4));
        history_ = static_cast<int*>(storage_.allocate(size_t(options_.context) * 4));
        check_cuda(cudaStreamSynchronize(stream_.p), "Dense service ready");
    }
    void command(const std::string& line) {
        std::istringstream in(line); std::string command, csv, extra; int slot;
        if (!(in >> command >> slot) || slot != 0) throw std::invalid_argument("Dense requires slot 0");
        if(command=="HEADPOLICY"||command=="SERVICEABORT") {
            int value=-1;
            if(!(in>>value)||in>>extra||active_||original_gemm_||weights_->head.quant!=Quant::Q6K||
               value<0||value>(command=="HEADPOLICY"?2:1))
                throw std::invalid_argument("Q36 cold policy requires idle original Q6 owner");
            graph_.reset();graph_cache_.invalidate(mtp_.get());
            check_cuda(cudaStreamSynchronize(stream_.p),"Q36 cold policy Graph drain");
            if(mtp_)mtp_abort::clear(*mtp_);
            if(command=="HEADPOLICY")set_q6_head_warps(value);
            else {
                if(value&&(options_.mtp!=2||options_.kv!="full"))
                    throw std::invalid_argument("MTP abort policy requires MTP2/full service");
                service_abort_=value!=0;
            }
            std::cout<<command<<" slot=0\n";return;
        }
        if(command=="ATTENTIONPOLICY") {
            std::string mode;
            if(!(in>>mode)||in>>extra||active_||(mode!="Q36_BASE"&&mode!="Q36_ONLINE_T3")||
               original_gemm_||weights_->head.quant!=Quant::Q6K)
                throw std::invalid_argument("Q36 attention policy requires idle original Q6 owner");
            if(mode=="Q36_ONLINE_T3"&&(options_.mtp!=2||options_.kv!="full"||
               spec_.n_head!=24||spec_.n_head_kv!=4||spec_.head_dim!=256))
                throw std::invalid_argument("Q36 online T3 requires MTP2/full GPU Q8 geometry");
            graph_.reset();graph_cache_.invalidate(mtp_.get());
            check_cuda(cudaStreamSynchronize(stream_.p),"Q36 attention policy Graph drain");
            q36_online_requested_=mode=="Q36_ONLINE_T3";
            std::cout<<"ATTENTIONPOLICY slot=0\n";return;
        }
        if(command=="SHAREDPREPARE") {
            int enabled;
            auto* provider=dynamic_cast<Q35SharedNvfp4*>(original_gemm_.get());
            if(!(in>>enabled)||in>>extra||active_||!provider||(enabled!=0&&enabled!=1))
                throw std::invalid_argument("Q35 shared prepare requires idle NVFP4 owner");
            graph_.reset();graph_cache_.invalidate(mtp_.get());
            check_cuda(cudaStreamSynchronize(stream_.p),"Q35 shared prepare graph drain");
            provider->shared_prepare(enabled!=0,stream_.p);
            std::cout<<"SHAREDPREPARE slot=0 epoch="<<provider->shared_epoch()<<"\n";return;
        }
        if(command=="MATRIXPOLICY") {
            std::string mode;
            if(!(in>>mode)||in>>extra||active_)throw std::invalid_argument("Matrix policy requires idle owner");
            if(auto* shared=dynamic_cast<Q35SharedNvfp4*>(original_gemm_.get());shared&&shared->shared_prepare()){
                graph_.reset();graph_cache_.invalidate(mtp_.get());
                check_cuda(cudaStreamSynchronize(stream_.p),"Q35 matrix policy shared graph drain");
                shared->shared_prepare(false,stream_.p);
            }
            const std::string requested_mode=mode;pp_prefetch_select(original_gemm_.get(),mode);pp_library_.select(mode);
            const bool tail=mode=="Q8A_P2TAIL"||mode=="Q8A_P2TAILS8";
            pp_s8_requested_=mode=="Q8A_P2S8"||mode=="Q8A_P2TAILS8";
            if(tail||pp_s8_requested_)mode="Q8A_P2VEC";
            if(mode=="Q6_BASE"||mode=="Q6_SOA") {
                if(original_gemm_||weights_->head.quant!=Quant::Q6K)throw std::invalid_argument("Q6 policy requires original Q6 Dense model");
                model_->select_q6_soa(mode=="Q6_SOA");std::cout<<"MATRIXPOLICY slot=0\n";return;
            }
            if(!original_gemm_||(mode!="CUBLAS"&&mode!="PACKED"&&mode!="CUTLASS_SELECT"&&mode!="CUTLASS_Q8"&&mode!="CUDA_SELECT"&&mode!="Q8A_NV"&&mode!="Q8A_ALL"&&mode!="Q8A_PAIR"&&mode!="Q8A_FAST"&&mode!="Q8A_SOA"&&mode!="Q8A_SOAFAST"&&mode!="Q8A_P2PP"&&mode!="Q8A_P2WIDE"&&mode!="Q8A_P2PAIR"&&mode!="Q8A_P2D8"&&mode!="Q8A_P2D8PAIR"&&mode!="Q8A_P2ONLINE"&&mode!="Q8A_P2NOCAP"&&mode!="Q8A_P2GDN"&&mode!="Q8A_P2TUNE"&&mode!="Q8A_P2FUSE"&&mode!="Q8A_P2FUSELT"&&mode!="Q8A_P2S16"&&mode!="Q8A_P2DUAL"&&mode!="Q8A_P2GCACHE"&&mode!="Q8A_P2VEC"&&mode!="Q8A_P2LONG"))
                throw std::invalid_argument("Dense matrix policy requires a registered idle GPU provider");
            auto* provider=static_cast<Nvfp4Gemm*>(original_gemm_.get());
            provider->packed_decode(true);provider->library_decode(mode=="CUBLAS");
            graph_cache_.select(mode=="Q8A_P2GCACHE"||mode=="Q8A_P2VEC"||mode=="Q8A_P2LONG",requested_mode);
            provider->q8_pp_vector(mode=="Q8A_P2VEC"||mode=="Q8A_P2LONG");
            provider->q8_pp_long(mode=="Q8A_P2LONG");provider->q8_pp_tail(tail);
            pp_s16_requested_=mode=="Q8A_P2S16";pp_dual_requested_=mode=="Q8A_P2DUAL"||graph_cache_.requested();
            pp_tune_requested_=mode=="Q8A_P2TUNE"||mode=="Q8A_P2FUSE"||mode=="Q8A_P2FUSELT"||pp_s16_requested_||pp_dual_requested_;
            provider->q8_pp_lt(mode=="Q8A_P2TUNE"||mode=="Q8A_P2FUSELT");
            provider->q8_pp_fused(mode=="Q8A_P2FUSE"||mode=="Q8A_P2FUSELT"||pp_s16_requested_||pp_dual_requested_);
            skip_pp_requested_=mode=="Q8A_P2NOCAP"||mode=="Q8A_P2GDN"||pp_tune_requested_;
            provider->q8_pp_gdn(mode=="Q8A_P2GDN"||pp_tune_requested_);
            online_requested_=mode=="Q8A_P2ONLINE"||skip_pp_requested_;
            const bool pp_pair=mode=="Q8A_P2D8PAIR"||online_requested_,decode8=mode=="Q8A_P2D8"||pp_pair,pair_half=mode=="Q8A_P2PAIR"||decode8,wide=mode=="Q8A_P2WIDE"||pair_half,pp_p2=mode=="Q8A_P2PP"||wide;
            const bool fastsoa=mode=="Q8A_SOAFAST"||pp_p2,soa=mode=="Q8A_SOA"||fastsoa;
            const bool fast=mode=="Q8A_FAST"||soa,pair=mode=="Q8A_PAIR"||fast;
            const int q8a=(mode=="Q8A_ALL"||pair)?2:mode=="Q8A_NV"?1:0;
            provider->q8_activation(q8a);provider->q8_pair(pair);provider->q8_fast(fast);provider->q8_soa(soa);
            provider->q8_fastsoa(fastsoa);provider->q8_pp_p2(pp_p2);wide_requested_=wide;provider->q8_head_wide(wide_requested_&&request_mtp_==2);provider->q8_pair_half(pair_half);provider->q8_pp_decode8(decode8);provider->q8_pp_pair(pp_pair);
            const bool fused=q8a||mode=="CUTLASS_SELECT"||mode=="CUTLASS_Q8"||mode=="CUDA_SELECT";
            provider->fused_cutlass(fused,false,0,fused);
            provider->q8_optimized(q8a||mode=="CUTLASS_Q8"||mode=="CUDA_SELECT");
            provider->mixed_decode(q8a||mode=="CUDA_SELECT");provider->q8_blocked(q8a||mode=="CUDA_SELECT");provider->q8_dp4a(0);provider->q8_warps(0);
            std::cout<<"MATRIXPOLICY slot=0\n";return;
        }
        if(command=="MTPPOLICY") {
            int value=-1;
            if(!(in>>value)||in>>extra||active_||(value!=0&&value!=options_.mtp))
                throw std::invalid_argument("Dense MTP policy must be0 or configured maximum, while idle");
            request_mtp_=value;
            if(original_gemm_)static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_head_wide(wide_requested_&&request_mtp_==2);
            std::cout<<"MTPPOLICY slot=0\n";return;
        }
        if(command=="PPPOLICY") {
            int enabled;
            if(!(in>>enabled)||in>>extra||active_||(enabled!=0&&enabled!=1)||
               (enabled&&(!reuse_eager_option_||(options_.mtp<1||options_.mtp>2)||options_.kv!="full")))
                throw std::invalid_argument("Invalid or unadmitted Dense PP policy");
            reuse_eager_request_=enabled!=0;std::cout<<"POLICY slot=0\n";return;
        }
        if(command=="INFO") {
            if(in>>extra||!active_)throw std::invalid_argument("Dense INFO requires an active request");
            const auto now=speculative_?mtp_->counters():request_counts_;
            std::cout<<"INFO actual_mtp="<<(speculative_?options_.mtp:0)
                <<" requested_mtp="<<options_.mtp<<" policy_mtp="<<request_mtp_
                <<" effective_mtp="<<(speculative_?options_.mtp:0)
                <<" sampling="<<sampling_.enabled()
                <<" mtp_sampling_fallback="<<(sampling_.enabled()&&request_mtp_>0&&!speculative_)
                <<" sampled_mtp="<<(speculative_&&mtp_->sampled())
                <<" cuda_graph="<<(speculative_?mtp_->graph_enabled():bool(graph_))
                <<" original_gpu_provider="<<bool(original_gemm_)
                <<" nvfp4_packed="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->packed_decode())
                <<" nvfp4_library="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->library_decode())
                <<" fused_decode="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->fused_decode())
                <<" fused_selective="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->fused_selective())
                <<" fused_recorded_calls="<<(original_gemm_?static_cast<Nvfp4Gemm*>(original_gemm_.get())->fused_calls():0)
                <<" kv_capacity="<<model_->cache_capacity()<<" online_min_capacity="<<(graph_cache_.requested()?8192:32768)
                <<" mtp_graph_cache="<<graph_cache_.requested()<<" mtp_graph_reused="<<graph_cache_.hit()
                <<" skip_pp_graph="<<skip_pp_requested_
                <<" prefix_graph_captured="<<(speculative_&&mtp_->prefix_graph_enabled())
                <<" prefix_graph_nodes="<<(speculative_?mtp_->prefix_graph_nodes():0)
                <<" mtp_graph_nodes="<<(speculative_?mtp_->graph_nodes():0)
                <<" q8_attention_online="<<online_requested_
                <<" q8_pp_tail="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_tail())
                <<" q8_pp_tail_calls="<<(original_gemm_?static_cast<Nvfp4Gemm*>(original_gemm_.get())->pp_tail_calls():0)
                <<" pp_library="<<pp_library_.selected()<<" pp_library_recorded_calls="<<pp_library_.recorded()
                <<" q8_pp_s8="<<pp_s8_requested_
                <<" q8_pp_long="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_long())
                <<" q8_pp_long_calls="<<(original_gemm_?static_cast<Nvfp4Gemm*>(original_gemm_.get())->pp_long_calls():0)
                <<" q8_pp_vector="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_vector())
                <<" q8_pp_dual="<<pp_dual_requested_
                <<" q8_pp_s16="<<pp_s16_requested_
                <<" q8_pp_tuned="<<pp_tune_requested_
                <<" q35_shared_prepare="<<(original_gemm_&&static_cast<Q35SharedNvfp4*>(original_gemm_.get())->shared_prepare())
                <<" q35_shared_epoch="<<(original_gemm_?static_cast<Q35SharedNvfp4*>(original_gemm_.get())->shared_epoch():0)
                <<" q35_shared_group_submissions="<<(original_gemm_?static_cast<Q35SharedNvfp4*>(original_gemm_.get())->shared_groups():0)
                <<" pp_prefetch_calls="<<pp_prefetch_recorded(original_gemm_.get())
                <<" pp_prefetch_builds="<<pp_prefetch_builds(original_gemm_.get())<<" pp_prefetch_hits="<<pp_prefetch_hits(original_gemm_.get())
                <<" q8_pp_fused="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_fused())
                <<" q8_pp_fused_calls="<<(original_gemm_?static_cast<Nvfp4Gemm*>(original_gemm_.get())->pp_fused_calls():0)
                <<" q8_pp_lt="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_lt())
                <<" q8_pp_lt_calls="<<(original_gemm_?static_cast<Nvfp4Gemm*>(original_gemm_.get())->pp_lt_calls():0)
                <<" q8_pp_lt_algorithm="<<(original_gemm_?static_cast<Nvfp4Gemm*>(original_gemm_.get())->pp_lt_info().algorithm:-1)
                <<" q8_pp_gdn="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_gdn())
                <<" q8_pp_pair="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_pair())
                <<" q8_pp_decode8="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_decode8())
                <<" q36_online_requested="<<q36_online_requested_<<" q36_online_selected="<<q36_online_selected_
                <<" q36_online_enqueues="<<q36_online::enqueues.online<<" q36_parent_enqueues="<<q36_online::enqueues.parent
                <<" head_warps="<<q6_head_warps()<<" service_abort_requested="<<service_abort_
                <<" q6_soa="<<model_->q6_soa_enabled()
                <<" q8_pair_half="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pair_half())
                <<" q8_head_wide="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_head_wide())
                <<" q8_pp_p2="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pp_p2())
                <<" q8_fastsoa="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_fastsoa())
                <<" q8_soa="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_soa())
                <<" q8_fast="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_fast())
                <<" q8_pair="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_pair())
                <<" q8_activation="<<(original_gemm_?static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_activation():0)
                <<" mixed_decode="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->mixed_decode())
                <<" q8_blocked="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_blocked())
                <<" q8_optimized="<<(original_gemm_&&static_cast<Nvfp4Gemm*>(original_gemm_.get())->q8_optimized())
                <<" original_calls_recorded="<<(original_gemm_?original_gemm_->calls():0)
                <<" rounds="<<now.rounds-request_counts_.rounds
                <<" proposed="<<now.proposed-request_counts_.proposed
                <<" accepted="<<now.accepted-request_counts_.accepted
                <<" target_rows="<<now.target_rows-request_counts_.target_rows
                <<" replayed_rows="<<now.replayed_rows-request_counts_.replayed_rows
                <<" D2H_bytes="<<now.D2H_bytes-request_counts_.D2H_bytes
                <<" D2D_bytes="<<now.D2D_bytes-request_counts_.D2D_bytes
                <<" completion_joins="<<now.completion_joins-request_counts_.completion_joins
                <<" H2D_bytes="<<now.H2D_bytes-request_counts_.H2D_bytes
                <<" graph_draft_replays="<<now.graph_draft-request_counts_.graph_draft
                <<" graph_verify_replays="<<now.graph_verify-request_counts_.graph_verify
                <<" graph_replay_replays="<<now.graph_replay-request_counts_.graph_replay
                <<" graph_rewind_replays="<<now.graph_rewind-request_counts_.graph_rewind
                <<" graph_commit_replays="<<now.graph_commit-request_counts_.graph_commit
                <<" graph_prefix_restore_replays="<<now.graph_prefix_restore-request_counts_.graph_prefix_restore
                <<" prefix_written_bytes="<<now.prefix_written_bytes-request_counts_.prefix_written_bytes
                <<" prefix_restored_rows="<<now.prefix_restored_rows-request_counts_.prefix_restored_rows
                <<" graph_PP_replays="<<now.graph_PP-request_counts_.graph_PP
                <<" pp_reused="<<reused_prefill_<<" pp_eager_us="<<pp_eager_us_
                <<" pp_capture_us="<<pp_capture_us_<<" pp_replay_us="<<pp_replay_us_
                <<'\n';
            // One bounded response at request end; never per-token diagnostics.
            return;
        }
        if (command == "RELEASE") {
            if (in >> extra) throw std::invalid_argument("unexpected RELEASE field");
            release(); std::cout << "RELEASED slot=0\n"; return;
        }
        if (command == "CONFIG") {
            if (active_) throw std::invalid_argument("Dense slot is active");
            sampling_.configure(in, options_.context);graph_cache_.configure(line); std::cout << "CONFIGURED slot=0\n"; return;
        }
        if (command == "PREPARE" || command == "PREPAREV") {
            int max_new;
            if (!(in >> max_new >> csv) || in >> extra || active_) throw std::invalid_argument("invalid Dense PREPARE");
            auto visual = command=="PREPAREV" ? read_visual_prompt(csv,options_.context,int(spec_.n_embd),int(spec_.n_vocab)) : nullptr;
            const auto prompt = visual ? visual->ids : parse_ids(csv, int(spec_.n_vocab));
            if (prompt.empty() || max_new < 1 || prompt.size() + size_t(max_new) > size_t(options_.context))
                throw std::invalid_argument("Dense prompt plus output exceeds context");
            graph_.reset();
            graph_cache_.prepare(mtp_.get(),int(prompt.size()),int(prompt.size())+max_new,request_mtp_,
                reuse_eager_request_&&skip_pp_requested_&&options_.kv=="full"&&command=="PREPARE"&&max_new>=3);
            model_->set_visual_input(std::move(visual),stream_.p);
            const int maximum = int(prompt.size()) + max_new;
            model_->reserve_cache(maximum, stream_.p);
            speculative_ = request_mtp_ > 0 && (!sampling_.enabled()||sampled_mtp_option_) && prompt.size() >= 2 && max_new >= 3;
            request_counts_=mtp_?mtp_->counters():MtpCounters{};
            reused_prefill_=false;pp_eager_us_=pp_capture_us_=pp_replay_us_=0;
            apply_service_profile(*model_, *shape_, int(prompt.size()), speculative_ ? options_.mtp : 0);
            if(pp_s8_requested_)model_->use_pp_gdn_kernel(gdn_recurrence_subwarp8_dual,gdn_subwarp16_workspace_bytes);
            else if(pp_dual_requested_)
                model_->use_pp_gdn_kernel(gdn_recurrence_subwarp16_dual,gdn_subwarp16_workspace_bytes);
            else if(pp_s16_requested_)
                model_->use_pp_gdn_kernel(gdn_recurrence_subwarp16,gdn_subwarp16_workspace_bytes);
            if(pp_tune_requested_){
                prefill_short32_configure();
                model_->attention_work().use_pp_attention_kernel(prefill_attention_tuned,48);
            }
            pp_library_.apply(*model_,*fallback_,storage_,spec_);
            if(online_requested_) {
                verify_online_configure();
                model_->attention_work().use_verify_attention_kernel(graph_cache_.requested()?
                    verify_online_dispatch<8192>:verify_online_dispatch<32768>);
            }
            q36_online_selected_=false;
            if(!original_gemm_)q36_online_selected_=q36_online::install(*model_,q36_online_requested_&&speculative_&&
                options_.mtp==2&&options_.kv=="full"&&command=="PREPARE"&&(prompt.size()==8192||prompt.size()==30720));
            if(original_gemm_)model_->use_original_gemm(original_gemm_.get());
            model_->reset_prefill(stream_.p); sampling_.prepare(prompt);
            check_cuda(cudaMemcpyAsync(input_, prompt.data(), prompt.size() * 4, cudaMemcpyHostToDevice, stream_.p), "Dense prompt payload");
            const auto start = Clock::now();
            if (speculative_) {
                if (!mtp_) {
                    mtp_ = std::make_unique<NativeMtp>(*model_, file_, stream_.p);
                    mtp_->use_gpu_draft_chain(true); mtp_->use_prefix_states(options_.mtp == 2);
                }
                if(!graph_cache_.hit())mtp_abort::select(*mtp_,stream_.p,service_abort_);
                if(!graph_cache_.hit())mtp_->configure_sampling(sampling_.enabled(),sampling_.params,input_,int(prompt.size()));
                mtp_->reserve_cache(maximum, stream_.p);
                const int tokens = int(prompt.size());
                const bool reuse=reuse_eager_request_&&(options_.mtp==1||options_.mtp==2)&&options_.kv=="full"&&command=="PREPARE";
                // Admit shapes with one full eager prefix, then retain that
                // valid state across construction of the continuation graphs.
                const int warm = reuse?tokens:(std::min)(tokens, model_->max_tokens() + tokens % model_->max_tokens());
                auto phase=Clock::now();
                mtp_->prefill(input_, warm, stream_.p);
                check_cuda(cudaStreamSynchronize(stream_.p), "Dense MTP shape preparation");
                pp_eager_us_=static_cast<long long>(elapsed(phase)*1000);
                phase=Clock::now();
                if(graph_cache_.hit())mtp_->use_graphs(true);
                else mtp_->enable_graphs(input_, int(prompt.size()), maximum, stream_.p,!(reuse&&skip_pp_requested_));
                graph_cache_.captured(mtp_.get());
                check_cuda(cudaStreamSynchronize(stream_.p),"Dense MTP graph preparation join");
                pp_capture_us_=static_cast<long long>(elapsed(phase)*1000);
                bool replay=!reuse;
                if(replay) {
                    phase=Clock::now();model_->reset_prefill(stream_.p);
                    const bool eager_replay=!mtp_->prefix_graph_enabled();
                    if(eager_replay)mtp_->use_graphs(false);
                    mtp_->prefill(input_,tokens,stream_.p);
                    if(eager_replay)mtp_->use_graphs(true);
                    check_cuda(cudaStreamSynchronize(stream_.p),"Dense MTP prefix replay join");
                    pp_replay_us_=static_cast<long long>(elapsed(phase)*1000);
                } else reused_prefill_=true;
            } else model_->prefill(input_, int(prompt.size()), stream_.p);
            pending_ = pick(); position_ = int(prompt.size()); remaining_ = max_new - 1;
            if (!speculative_ && remaining_ && position_ <= 262142)
                graph_ = std::make_unique<ServiceDecodeGraph>(*model_, position_, maximum, stream_.p);
            active_ = true;
            std::cout << "PREFILL anchor=" << pending_ << " tokens=" << prompt.size()
                      << " pp_ms=" << elapsed(start) << " aux=0 cache_slots=0 native_info=1\n";
            return;
        }
        if (command == "STEP") {
            int anchor, requested;
            if (!(in >> anchor >> requested) || in >> extra || !active_ || anchor != pending_ || requested < 1 ||
                requested > remaining_ || position_ >= options_.context)
                throw std::invalid_argument("invalid Dense STEP or pending anchor");
            const auto start = Clock::now(); std::vector<int> picks;
            if (speculative_) picks = mtp_->round(anchor, position_, (std::min)(options_.mtp, requested - 1), stream_.p);
            else {
                sampling_.remember(anchor);
                if (graph_) graph_->launch(position_); else model_->decode(position_, stream_.p);
                picks.push_back(pick());
            }
            if (picks.empty() || int(picks.size()) > requested) throw std::runtime_error("Dense output extent");
            position_ += int(picks.size()); remaining_ -= int(picks.size()); pending_ = picks.back();
            std::cout << "STEPPED next=" << pending_ << " ms=" << elapsed(start) << " ids=";
            for (size_t i = 0; i < picks.size(); ++i) std::cout << (i ? "," : "") << picks[i];
            std::cout << '\n'; return;
        }
        throw std::invalid_argument("unsupported Dense command");
    }
    void recover() { release(); }
};
} // namespace
int strata_qwen_v100_completed_serve(int argc, char** argv) {
    try {
        const auto options = strata::completed_v100::models::NativeOptions::parse(argc, argv);
        if(options.family!="qwen35_dense")throw std::invalid_argument("Qwen service requires the Dense family");
        DenseService service(options);
        std::cout << "READY slots=1\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            try {
                if (line.size() > size_t(options.context) * 24 + 1024) throw std::invalid_argument("Dense command too long");
                service.command(line);
            } catch (const std::exception& e) {
                service.recover(); std::cout << "ERROR " << e.what() << '\n';
            }
            std::cout << std::flush;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "Dense service: " << e.what() << '\n'; return 1; }
}
