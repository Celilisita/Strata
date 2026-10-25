#include "gdn_long_chunk_policy.hpp"
#include "strata/models/native_options.hpp"
#include "strata/models/request_sampling.hpp"
#include "strata/qwen35/native/mtp.hpp"
#include "service_profile.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include "q8_pp_sgemm.hpp"
#include "q36_startup_body_plan.hpp"
#include "q36_nextn_nvfp4_policy.hpp"
#include "q36_ab_nvfp4_policy.hpp"
#include "q35_four_projection_policy.hpp"
#include "strata/qwen35/native/request_layout.hpp"
#include "strata/qwen35/native/cold_head_binding.hpp"
#include "cold_head_environment.hpp"
#include "gdn_output_elision_policy.hpp"
#include "fp4_compensated_service.hpp"
#include "mtp_exec_policy.hpp"
#include "ffn_down_add.hpp"
#include "tiny_parameter_pair_policy.hpp"
#include "tiny_parameter_pair.hpp"
#include "service_decode_graph.hpp"
#include "./service_mtp_prefill.hpp"
#include "service_ram_request.hpp"
#include "fresh_pool_policy.hpp"
#include "strata/qwen35/native/mtp_sampling.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace {
using namespace strata::qwen35::native;
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
    const strata::models::NativeOptions options_;
    strata::GgufFile file_;
    std::unique_ptr<strata::GgufFile> head_file_;
    strata::qwen35::Spec spec_;
    Stream stream_;
    std::unique_ptr<DenseWeights> weights_;
    std::unique_ptr<DenseGemm> fallback_, gemm_, long_gemm_, partial_gemm_;
    std::unique_ptr<FfnShapePolicy> shape_;
    std::unique_ptr<DenseGemm> nvfp4_;
    std::unique_ptr<DenseModel> model_;
    DeviceStorage storage_;
    int* input_ = nullptr;
    int* history_ = nullptr;
    std::unique_ptr<strata::kernels::RadixSampler> sampler_;
    strata::models::RequestSampling sampling_;
    std::unique_ptr<MtpSampling> mtp_sampling_;
    std::unique_ptr<NativeMtp> mtp_;
    std::unique_ptr<ServiceDecodeGraph> graph_;
    int position_ = -1, pending_ = -1, remaining_ = 0;
    bool active_ = false, speculative_ = false, reuse_prefill_ = false, startup_mtp_ = false;
    std::string sampling_key_="unconfigured";
    // Cross-request prefix resume bookkeeping. The snapshot holds every token
    // whose KV row and recurrent state the previous request left behind.
    std::vector<int> snapshot_;
    bool resumable_ = false, visual_request_ = false;
    double ensure_mtp() {
        if (mtp_) return 0;
        const auto start = Clock::now();
        auto prepared = std::make_unique<NativeMtp>(*model_, file_, stream_.p, options_.mtp);
        prepared->use_gpu_draft_chain(true);
        prepared->use_prefix_states(options_.mtp >= 2);
        mtp_ = std::move(prepared);
        return elapsed(start);
    }
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
    void release(bool keep_state=false) {
        // Graphs borrow KV, prefix input and sampler state; drain them first.
        graph_.reset();
        if (mtp_) { mtp_->park_graphs(); mtp_->use_sampling(nullptr); }
        check_cuda(cudaStreamSynchronize(stream_.p), "Dense request drain");
        model_->set_visual_input(nullptr,stream_.p);
        if (keep_state && !visual_request_ && position_ > 0) {
            // The snapshot and the recurrent/KV state stop at the same position,
            // so both stay authoritative for the next request's prefix.
            snapshot_.resize(size_t(position_));
            resumable_ = true;
        } else {
            model_->reset_prefill(stream_.p);
            snapshot_.clear(); resumable_ = false;
        }
        visual_request_ = false;
        mtp_exec_stats() = {};
        active_ = speculative_ = false; position_ = pending_ = -1;
        remaining_ = 0; sampling_ = {}; sampling_key_="unconfigured";
    }
public:
    explicit DenseService(strata::models::NativeOptions o) : options_(std::move(o)), file_(options_.model) {
        std::string error;
        if (!strata::qwen35::load_spec(file_, spec_, error)) throw std::runtime_error(error);
        if (options_.context != int(spec_.context_length))
            throw std::invalid_argument("Dense native profile requires the model context length");
        const bool qwen35_fp4=options_.variant=="qwen35_27b_nvfp4";
        configure_q35_four_projection_profile(qwen35_fp4);
        const bool q36_nextn=configure_q36_nextn_nvfp4(options_.variant,options_.mtp);
        const bool q36_ab=configure_q36_ab_nvfp4(options_.variant,q36_nextn);
        if(q36_nextn)validate_q36_nextn_nvfp4_asset(file_,spec_,q36_ab);
        if(qwen35_fp4){
            if(std::getenv("STRATA_NVFP4_HEAD_FILE"))
                throw std::invalid_argument("Qwen3.5 derived NVFP4 requires its embedded shared head");
            validate_qwen35_nvfp4_asset(spec_);
        }
        ColdHeadBinding head_binding;
        if(const char *head_path=std::getenv("STRATA_NVFP4_HEAD_FILE")){
            if(!*head_path)throw std::invalid_argument("empty NVFP4 head file");
            head_file_=std::make_unique<strata::GgufFile>(head_path);
            head_binding = cold_head_binding_from_environment(head_file_.get());
            if (options_.variant == "qwen36_27b" && !head_binding.has_source_identity())
                throw std::invalid_argument("Qwen3.6 NVFP4 head requires its explicit conversion identity");
            validate_cold_head_binding(file_,spec_,head_binding);
        }
        if (options_.variant == "qwen38_27b_uncensored_nvfp4" && !head_binding.fixture)
            throw std::invalid_argument("uncensored NVFP4 variant requires its bound head file");
        const char* startup_control = std::getenv("STRATA_DENSE_MTP_STARTUP");
        const std::string startup_mode = startup_control ? startup_control : "0";
        if (startup_mode != "0" && startup_mode != "1")
            throw std::invalid_argument("Dense MTP startup option must be0or1");
        startup_mtp_ = startup_mode == "1";
        if (startup_mtp_ && (options_.variant != "qwen36_27b" || options_.mtp < 1))
            throw std::invalid_argument("Dense MTP startup requires Qwen3.6 with MTP enabled");
        stream_.open();
        reuse_prefill_ = reuse_service_mtp_prefill();
        CompensatedFfnLoadScope compensated_load(options_.mtp);
        weights_ = std::make_unique<DenseWeights>(file_, spec_, stream_.p, false, head_binding);
        if((qwen35_fp4||q36_nextn)&&options_.mtp>0)weights_->prepare_mtp_nvfp4(file_,spec_,stream_.p);
        if (options_.variant == "qwen38_27b_uncensored_nvfp4" &&
            (weights_->fp4_weights.size() <= 1 || weights_->head.quant != Quant::NVFP4))
            throw std::invalid_argument("uncensored NVFP4 variant requires converted main weights and head");
        fallback_ = make_cublas_pair_gemm(stream_.p);
        gemm_ = make_cutlass_swiglu_gemm(*fallback_, stream_.p);
        const auto caps = capabilities(0);
        if (caps.major == 12 && caps.minor == 0) {
            long_gemm_ = make_cutlass_swiglu_gemm(*fallback_, stream_.p, 6);
            const char *partial = std::getenv("STRATA_DENSE_SM120_PARTIAL16");
            if (partial && partial[0] == '1') partial_gemm_ = make_partial16_gemm(*fallback_, stream_.p);
        }
        shape_ = std::make_unique<FfnShapePolicy>(*gemm_, 8192,
            partial_gemm_ ? partial_gemm_.get() : long_gemm_.get(), partial_gemm_.get());
        model_ = std::make_unique<DenseModel>(spec_, *weights_, service_limits(options_.context, options_.kv == "ram8k-reuse"), stream_.p);
        if(!weights_->fp4_weights.empty())nvfp4_=make_nvfp4_runtime(*shape_,weights_->fp4_weights,
            model_->max_tokens(),int(spec_.n_ff),stream_.p,
            weights_->head.quant==Quant::NVFP4?weights_->head:Matrix{});
        if(!q36_ab)nvfp4_configure_q8_pp(nvfp4_?*nvfp4_:*shape_,options_.variant,spec_,*weights_,stream_.p);
        if(q36_ab)std::cerr<<"DENSE_Q36_AB_NVFP4 enabled=1 matrices=96 fp4_inventory="<<weights_->fp4_weights.size()
            <<" four_projection_min_tokens=5 tiny_independent=1 q8_sgemm_aux=0 owner=existing_Runtime\n";
        const auto q8=nvfp4_q8_pp_stats(nvfp4_?*nvfp4_:*shape_);
        if(q8.resident_bytes)std::cerr<<"DENSE_Q8_PP_SGEMM threshold=256 pairs=48 cache_bytes="<<q8.cache_bytes
            <<" workspace_bytes="<<q8.workspace_bytes<<" scratch_bytes="<<q8.scratch_bytes
            <<" resident_bytes="<<q8.resident_bytes<<" cold_transient_bytes="<<q8.cold_transient_bytes
            <<" cold_ms="<<q8.cold_ms<<" free_before="<<q8.free_before<<" free_after="<<q8.free_after<<'\n';
        input_ = static_cast<int*>(storage_.allocate(size_t(options_.context) * 4));
        history_ = static_cast<int*>(storage_.allocate(size_t(options_.context) * 4));
        if (q36_startup_body_plan_requested(options_.variant, options_.kv))
            nvfp4_plan_q36_body_8192(nvfp4_ ? *nvfp4_ : *shape_, spec_, stream_.p);
        if (startup_mtp_) {
            size_t free_before = 0, free_after = 0, total = 0;
            check_cuda(cudaMemGetInfo(&free_before, &total), "Dense startup MTP initial memory");
            const auto start = Clock::now();
            ensure_mtp();
            check_cuda(cudaStreamSynchronize(stream_.p), "Dense startup MTP complete");
            check_cuda(cudaMemGetInfo(&free_after, &total), "Dense startup MTP final memory");
            if (free_after < DeviceStorage::reserve_bytes())
                throw std::runtime_error("Dense startup MTP lacks configured device reserve");
            std::cerr << "DENSE_MTP_STARTUP enabled=1 setup_ms=" << elapsed(start)
                      << " device_owned_bytes=" << mtp_->bytes()
                      << " borrowed_bytes=" << mtp_->borrowed_bytes()
                      << " host_kv_bytes=" << mtp_->ram_bytes()
                      << " free_before=" << free_before << " free_after=" << free_after
                      << " graph_nodes=" << mtp_->graph_nodes() << " prefix_tokens=0"
                      << " CUDA_MODULE_LOADING=" << (std::getenv("CUDA_MODULE_LOADING") ?
                          std::getenv("CUDA_MODULE_LOADING") : "UNSET") << '\n';
        }
        check_cuda(cudaStreamSynchronize(stream_.p), "Dense service ready");
    }
    void command(const std::string& line) {
        std::istringstream in(line); std::string command, csv, extra; int slot;
        if (!(in >> command >> slot) || slot != 0) throw std::invalid_argument("Dense requires slot 0");
        if (command == "RELEASE") {
            if (in >> extra) throw std::invalid_argument("unexpected RELEASE field");
            release(true); std::cout << "RELEASED slot=0\n"; return;
        }
        if (command == "CONFIG") {
            if (active_) throw std::invalid_argument("Dense slot is active");
            sampling_.configure(in, options_.context); sampling_key_=line; std::cout << "CONFIGURED slot=0\n"; return;
        }
        if (command == "PREPARE" || command == "PREPAREV") {
            int max_new;
            if (!(in >> max_new >> csv) || in >> extra || active_) throw std::invalid_argument("invalid Dense PREPARE");
            auto visual = command=="PREPAREV" ? read_visual_prompt(csv,options_.context,int(spec_.n_embd),int(spec_.n_vocab)) : nullptr;
            const auto prompt = visual ? visual->ids : parse_ids(csv, int(spec_.n_vocab));
            if (prompt.empty() || max_new < 1 || prompt.size() + size_t(max_new) > size_t(options_.context))
                throw std::invalid_argument("Dense prompt plus output exceeds context");
            // Reuse the previous request's trailing state only when the new
            // prompt repeats that whole sequence: the linear recurrence cannot
            // rewind to a shorter common prefix, so anything less is a fresh run.
            int resume_tokens = 0;
            if (!visual && resumable_ && !snapshot_.empty()) {
                const size_t limit = std::min(snapshot_.size(), prompt.size());
                size_t common = 0;
                while (common < limit && snapshot_[common] == prompt[common]) ++common;
                if (common == snapshot_.size() && common < prompt.size()) resume_tokens = int(common);
                else if (model_->checkpoint_position() > 0 && size_t(model_->checkpoint_position()) <= common)
                    resume_tokens = model_->checkpoint_position();
            }
            const bool resume = resume_tokens > 0;
            begin_phase_kv_request();
            // Give the configured reserve back before any fresh-request setup:
            // the previous long prefix may still hold its owned PP staging.
            model_->attention_work().release_staging();
            const int maximum = int(prompt.size()) + max_new;
            const bool layout_change=nvfp4_&&nvfp4_request_layout_change_required(*nvfp4_,int(prompt.size()));
            graph_.reset();
            if(mtp_){mtp_->park_graphs();if(layout_change)mtp_->release_graphs();}
            if(nvfp4_)nvfp4_prepare_request_layout(*nvfp4_,int(prompt.size()),layout_change,stream_.p);
            if(mtp_)mtp_->prepare_tg_graph_reuse(sampling_key_,int(prompt.size()),maximum,
                !visual&&!resume&&options_.mtp==2&&reuse_prefill_&&skip_service_mtp_prefix_graph(reuse_prefill_),stream_.p);
            model_->set_visual_input(std::move(visual),stream_.p);
            const char* preserve_old=std::getenv("STRATA_DENSE_FRESH_KV_GROWTH");
            if(resume)model_->reserve_cache(maximum,stream_.p);
            else if(preserve_old&&preserve_old[0]=='0')model_->reserve_cache(maximum,stream_.p);
            else model_->reserve_fresh_cache(maximum,stream_.p);
            speculative_ = options_.mtp > 0 && prompt.size() >= 2 && max_new >= options_.mtp + 1;
            const char* ram_control=std::getenv("STRATA_DENSE_RAM_REQUEST_BOUND");
            const int ram_bound=ram_control&&ram_control[0]=='0'?0:resident_ram_request_bound(
                maximum,speculative_?options_.mtp:0,model_->kv_resident(),model_->cache_logical_capacity());
            model_->attention_work().use_ram_request_bound(ram_bound);
            const char *prefix_control=std::getenv("STRATA_DENSE_RAM_PREFIX_BOUND");
            const int prefix_bound=prefix_control&&prefix_control[0]=='0'?0:resident_ram_request_bound(
                int(prompt.size()),0,model_->kv_resident(),model_->cache_logical_capacity());
            model_->attention_work().use_ram_prefix_bound(prefix_bound);
            apply_service_profile(*model_, nvfp4_?*nvfp4_:*shape_, int(prompt.size()), speculative_ ? options_.mtp : 0, options_.variant=="qwen36_27b");
            const char *down_control=std::getenv("STRATA_NVFP4_DOWN_ADD");
            nvfp4_configure_down_add_request(nvfp4_?*nvfp4_:*shape_,nvfp4_&&(!down_control||down_control[0]!='0'));
            const char *pair_control=std::getenv("STRATA_DENSE_PARAM_PAIR");
            const bool pair_enabled=nvfp4_&&(!pair_control||pair_control[0]!='0');
            if(pair_enabled){const auto r=tiny_batch_parameter_pair_resources();
                if(r.local_bytes||r.blocks<1)throw std::runtime_error("parameter pair resources unavailable");}
            configure_tiny_parameter_pair_request(pair_enabled);
            configure_gdn_output_elision_request(nvfp4_!=nullptr,!model_->visual_positions());
            nvfp4_configure_head_request(nvfp4_?*nvfp4_:*shape_,true);
            if(!resume)model_->reset_prefill(stream_.p);
            else{model_->restore_checkpoint(stream_.p);model_->resume_kv(resume_tokens,stream_.p);}
            sampling_.prepare(prompt);
            check_cuda(cudaMemcpyAsync(input_, prompt.data(), prompt.size() * 4, cudaMemcpyHostToDevice, stream_.p), "Dense prompt payload");
            const auto start = Clock::now();
            int evaluated = int(prompt.size());
            double mtp_setup_ms = 0;
            if (speculative_) {
                if (!mtp_) {
                    const auto setup = Clock::now();
                    mtp_ = std::make_unique<NativeMtp>(*model_, file_, stream_.p, options_.mtp);
                    mtp_->use_gpu_draft_chain(true); mtp_->use_prefix_states(options_.mtp >= 2);
                    mtp_->prepare_tg_graph_reuse(sampling_key_,int(prompt.size()),maximum,
                        !model_->visual_positions()&&options_.mtp==2&&reuse_prefill_&&skip_service_mtp_prefix_graph(reuse_prefill_),stream_.p);
                    mtp_setup_ms = elapsed(setup);
                } else if (startup_mtp_ && mtp_->refresh_request_attention_profile()) {
                    // Only the opt-in startup owner refreshes its profile;
                    // existing lazy owners retain their original request route.
                    mtp_->prepare_tg_graph_reuse(sampling_key_,int(prompt.size()),maximum,
                        !model_->visual_positions()&&options_.mtp==2&&reuse_prefill_&&skip_service_mtp_prefix_graph(reuse_prefill_),stream_.p);
                }
                if (sampling_.enabled()) {
                    if (!mtp_sampling_) mtp_sampling_ = std::make_unique<MtpSampling>(int(spec_.n_vocab));
                    const bool phase_paged=phase_kv_will_demote(int(prompt.size()),maximum,!model_->visual_positions(),options_.mtp,
                        reuse_prefill_&&skip_service_mtp_prefix_graph(reuse_prefill_));
                    mtp_sampling_->configure(sampling_.params, sampling_.history, stream_.p,phase_paged?0:ram_bound);
                    mtp_->use_sampling(mtp_sampling_.get());
                } else mtp_->use_sampling(nullptr);
                const char *fresh_mtp=std::getenv("STRATA_DENSE_MTP_FRESH_KV_GROWTH");
                if(resume||!(fresh_mtp?fresh_mtp[0]!='0':fresh_mtp_default(maximum)))mtp_->reserve_cache(maximum,stream_.p);
                else mtp_->reserve_fresh_cache(maximum,stream_.p);
                mtp_->use_ram_request_bound(ram_bound);
                evaluated = prepare_service_mtp_prefix(*mtp_, *model_, input_,
                    int(prompt.size()), maximum, reuse_prefill_, stream_.p, resume_tokens);
            } else model_->prefill(input_ + resume_tokens, int(prompt.size()) - resume_tokens, stream_.p, resume_tokens);
            if(!speculative_) evaluated = int(prompt.size()) - resume_tokens;
            // Prefill staging is dead once the prefix is complete; return it so
            // decode and the next request keep the full configured reserve.
            model_->attention_work().release_staging();
            pending_ = pick(); position_ = int(prompt.size()); remaining_ = max_new - 1;
            if (!speculative_ && remaining_ && position_ <= 262142)
                graph_ = std::make_unique<ServiceDecodeGraph>(*model_, position_, maximum, stream_.p);
            active_ = true;
            snapshot_.assign(prompt.begin(),prompt.end());
            resumable_ = false; visual_request_ = visual != nullptr;
            std::cout << "PREFILL anchor=" << pending_ << " tokens=" << prompt.size()
                      << " pp_ms=" << elapsed(start) << " speculative=" << int(speculative_)
                      << " pp_evaluated=" << evaluated
                      << " pp_resume=" << resume_tokens
                      << " pp_reused=" << int(speculative_ && reuse_prefill_)
                      << " mtp_setup_ms=" << mtp_setup_ms
                      << " graph_capture_ms=" << (speculative_ ? mtp_->graph_cold_ms() : 0)
                      << " ram_resident_rows=" << model_->kv_resident()
                      << " ram_logical_capacity=" << model_->cache_logical_capacity()
                      << " q8_pp_sgemm_host_submissions=" << nvfp4_q8_pp_stats(nvfp4_?*nvfp4_:*shape_).host_submissions
                      << " q8_pp_sgemm_eager_shapes=" << nvfp4_q8_pp_stats(nvfp4_?*nvfp4_:*shape_).eager_shapes
                      << " aux=0 cache_slots=0\n";
            return;
        }
        if (command == "STEP") {
            int anchor, requested;
            if (!(in >> anchor >> requested) || in >> extra || !active_ || anchor != pending_ || requested < 1 ||
                requested > remaining_ || position_ >= options_.context)
                throw std::invalid_argument("invalid Dense STEP or pending anchor");
            const auto start = Clock::now(); std::vector<int> picks;
            if (speculative_) {
                if (sampling_.enabled()) mtp_sampling_->begin(sampling_.params.counter, sampling_.history, stream_.p);
                picks = mtp_->round(anchor, position_, (std::min)(options_.mtp, requested - 1), stream_.p);
                if (sampling_.enabled()) {
                    sampling_.remember(anchor);
                    for (size_t i = 0; i+1 < picks.size(); ++i) sampling_.remember(picks[i]);
                    sampling_.params.counter += picks.size();
                }
            }
            else {
                sampling_.remember(anchor);
                if (graph_) graph_->launch(position_); else model_->decode(position_, stream_.p);
                picks.push_back(pick());
            }
            if (picks.empty() || int(picks.size()) > requested) throw std::runtime_error("Dense output extent");
            snapshot_.insert(snapshot_.end(),picks.begin(),picks.end());
            position_ += int(picks.size()); remaining_ -= int(picks.size()); pending_ = picks.back();
            std::cout << "STEPPED next=" << pending_ << " ms=" << elapsed(start)
                      << " proposed=" << (speculative_ ? (std::min)(options_.mtp, requested - 1) : 0)
                      << " accepted=" << (speculative_ ? int(picks.size()) - 1 : 0) << " ids=";
            for (size_t i = 0; i < picks.size(); ++i) std::cout << (i ? "," : "") << picks[i];
            std::cout << '\n'; return;
        }
        throw std::invalid_argument("unsupported Dense command");
    }
    void recover() {
        if(phase_kv_stats().failed)throw std::runtime_error("phase KV migration failed; reload required");
        release();
    }
};
} // namespace
int strata_qwen35_dense_serve(int argc, char** argv) {
    try {
        const auto options = strata::models::NativeOptions::parse(argc, argv);
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
