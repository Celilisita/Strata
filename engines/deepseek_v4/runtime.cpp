#include "runtime.hpp"
#include "../glm5_next/model_contract.hpp"
#include "../glm5_next/mtp_options.hpp"
#include "../glm5_next/verify4_options.hpp"
#include "../glm5_next/mtp_session.hpp"
#include "../glm5_next/cpu_placement.hpp"
#include "runtime_memory.hpp"
#include "mtp_sidecar.hpp"
#include "mtp_drafter.hpp"
#include "native_tg_graph.hpp"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-moe-cache.h"
#include "pp_stream_graph.hpp"
#include "device_cache.hpp"
#include "cache_leaf_aliases.hpp"
#include "expert_staging_attach.hpp"
#include "cold_expert_attach.hpp"
#include "expert_page_warm.hpp"

#include "gguf.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <iostream>
#include <cstdio>
#include <future>
#include <stdexcept>

#ifndef STRATA_DEEPSEEK_LLAMA_COMMIT
#error "The canonical source commit must be supplied by the independent CMake project"
#endif

namespace strata::deepseek_v4 {
namespace {

// Match only routed experts. Shared experts and router stay with dense weights.
constexpr const char* routed_experts = "blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight";

struct ExpertWarmPlan {std::vector<ExpertWarmRange> read,lock;};
ExpertWarmPlan expert_warm_plan(const llama_model& model,uint64_t source_budget){
    if(model.arch==LLM_ARCH_GLM5_NEXT)return strata::glm5_next::warm_plan<ExpertWarmPlan>(model);
    const char* priority=std::getenv("STRATA_DSV4_MTP_PIN_PRIORITY");
    const bool mtp_first=model.layers.size()==44&&priority&&!std::strcmp(priority,"1");
    const uint64_t original_budget=source_budget;
    size_t mtp_bytes=0;
    if(mtp_first)
        for(auto* tensor:{model.layers[43].ffn_gate_exps,model.layers[43].ffn_up_exps,model.layers[43].ffn_down_exps})
            mtp_bytes+=ggml_nbytes(tensor);
    if(mtp_first)source_budget=source_budget>mtp_bytes?source_budget-mtp_bytes:0;
    ExpertWarmPlan plan;uint64_t passed=0,total=0;
    for(size_t layer=0;layer<std::min<size_t>(43,model.layers.size());++layer)
        for(auto* tensor:{model.layers[layer].ffn_gate_exps,model.layers[layer].ffn_up_exps,model.layers[layer].ffn_down_exps}){
            if(!tensor||!tensor->data)throw std::runtime_error("expert page warm source absent");
            const size_t bytes=ggml_nbytes(tensor);
            const ExpertWarmRange range{static_cast<const uint8_t*>(tensor->data),bytes};
            plan.read.push_back(range);
            if(passed+bytes>source_budget)plan.lock.push_back(range);
            total+=bytes;
            passed+=bytes;
        }
    if(model.layers.size()==44)
        for(auto* tensor:{model.layers[43].ffn_gate_exps,model.layers[43].ffn_up_exps,model.layers[43].ffn_down_exps}){
            if(!tensor||!tensor->data)throw std::runtime_error("MTP expert page warm source absent");
            const size_t bytes=ggml_nbytes(tensor);const ExpertWarmRange range{static_cast<const uint8_t*>(tensor->data),bytes};
            plan.read.push_back(range);if(!mtp_first||mtp_bytes>original_budget)plan.lock.push_back(range);total+=bytes;
        }
    // Keep the complete source inventory available for an explicit full warm.
    // The default load path touches only the unregistered suffix, because
    // reading the complete GGUF adds substantial load time on this machine.
    if(total>(160ull<<30))throw std::runtime_error("expert page warm plan exceeds160GiB bound");
    return plan;
}

bool check_architecture(const std::string& path, std::string& error) {
    const gguf_init_params params{true, nullptr};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> header(
        gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!header) {
        error = "cannot read GGUF metadata: " + path;
        return false;
    }
    const int64_t key = gguf_find_key(header.get(), "general.architecture");
    if (key < 0 || gguf_get_kv_type(header.get(), key) != GGUF_TYPE_STRING ||
        !strata::glm5_next::accepted_architecture(gguf_get_val_str(header.get(), key))) {
        error = "GGUF architecture must be deepseek4 or glm5-next; GLM requires the matched Shard_Rewrite metadata";
        return false;
    }
    return true;
}



}  // namespace

Runtime::Runtime(const Options& options):options_(options){}

std::unique_ptr<Runtime> Runtime::load(const Options& options, std::string& error,
                                     ggml_backend_sched_eval_callback callback, void* data) {
    if (!validate_options(options, error)) return nullptr;
    auto runtime = std::unique_ptr<Runtime>(new Runtime(options));
    try {
        if (!runtime->initialize(error, callback, data)) return nullptr;
    } catch (const std::exception& failure) {
        error = failure.what();
        return nullptr;
    }
    return runtime;
}

bool Runtime::initialize(std::string& error, ggml_backend_sched_eval_callback callback, void* data) {
    if(packed_experts_requested()&&expert_source_private_mapping_requested())
        throw std::runtime_error("packed experts require read-only GGUF mappings; private mappings reserve the entire file in Windows commit");
    if (expert_source_pin_budget() && !options_.pinned_experts)
        throw std::runtime_error("source pinning requires pinned expert transport");
    if (!check_architecture(options_.model, error)) return false;
    configure_device_cache_process();
    static std::once_flag backend_init;
    std::call_once(backend_init, [] {
        // Match the normal CLI verbosity. Keep fatal/error output synchronous.
        llama_log_set([](ggml_log_level level,const char* message,void*) {
            const char* debug=std::getenv("STRATA_DSV4_DEBUG_LOG");
            if(level==GGML_LOG_LEVEL_DEBUG&&(!debug||std::strcmp(debug,"1")))return;
            std::fputs(message,stderr);if(level==GGML_LOG_LEVEL_ERROR)std::fflush(stderr);
        },nullptr);
        llama_backend_init();
    });

    devices_[0] = ggml_backend_dev_by_name(options_.device.c_str());
    if (!devices_[0]) {
        error = "specified CUDA device is unavailable: " + options_.device;
        return false;
    }
    const auto registration = ggml_backend_dev_backend_reg(devices_[0]);
    if (!registration || std::strcmp(ggml_backend_reg_name(registration), "CUDA") != 0) {
        error = "specified device does not belong to the CUDA backend";
        return false;
    }
    if (!llama_supports_mmap()) {
        error = "this platform lacks mmap required for host-resident expert weights";
        return false;
    }
    const auto cpu_buffer = ggml_backend_cpu_buffer_type();
    if (!cpu_buffer) {
        error = "the canonical CPU buffer backend is unavailable";
        return false;
    }
    overrides_[0] = {routed_experts, cpu_buffer};
    overrides_[1] = {nullptr, nullptr};

    auto model_params = llama_model_default_params();
    model_params.devices = devices_.data();
    model_params.tensor_buft_overrides = overrides_.data();
    model_params.n_gpu_layers = options_.gpu_layers;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    // With an explicit one-device list this is index zero, including CUDA1 etc.
    model_params.main_gpu = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.lazy_mode = LLAMA_LAZY_MODE_ON;
    model_params.use_extra_bufts = false;
    model_params.load_mtp = strata::glm5_next::mtp_requested();
    runtime_memory("before-model-load");
    model_ = strata::glm5_next::load(options_.model, model_params);
    runtime_memory("after-model-load");
    if (!model_) {
        error = "canonical DeepSeek4 model loading failed";
        return false;
    }
    if (!strata::glm5_next::accepted_architecture(metadata("general.architecture").c_str())) {
        error = "loaded model architecture differs from the validated GGUF";
        return false;
    }
    if (!strata::glm5_next::validate(*model_, error)) return false;
    vocab_ = llama_model_get_vocab(model_);
    glm_mtp_=model_->arch==LLM_ARCH_GLM5_NEXT&&strata::glm5_next::mtp_requested();
    if(const char* path=std::getenv("STRATA_DSV4_MTP_PATH")) {
        if(!device_expert_cache_requested()||!native_tg_cpu_only())throw std::runtime_error("MTP requires native CPU miss service and unified device cache");
        mtp_loading_backend_=ggml_backend_dev_init(devices_[0],nullptr);if(!mtp_loading_backend_)throw std::runtime_error("MTP weight backend initialization failed");
        mtp_weights_=std::make_unique<MtpSidecar>(path,mtp_loading_backend_);
        model_->strata_append_mtp_experts(mtp_weights_->tensor("blk.43.ffn_gate_exps.weight"),mtp_weights_->tensor("blk.43.ffn_up_exps.weight"),mtp_weights_->tensor("blk.43.ffn_down_exps.weight"));
    }

    return initialize_context(error, callback, data);
}

bool Runtime::initialize_context(std::string& error, ggml_backend_sched_eval_callback callback, void* data) {
    std::future<std::unique_ptr<ExpertPageWarm>> warming;
    const char* warm=std::getenv("STRATA_DSV4_WARM_UNPINNED");
    if(!page_warm_&&options_.pinned_experts&&warm&&!std::strcmp(warm,"1")){
        auto plan=expert_warm_plan(*model_,expert_source_pin_budget());
        const char* all=std::getenv("STRATA_DSV4_WARM_ALL_EXPERTS");
        if(!all||std::strcmp(all,"1"))plan.read=plan.lock;
        warming=std::async(std::launch::async,[plan=std::move(plan)]()mutable{
            auto owner=std::make_unique<ExpertPageWarm>(std::move(plan.read),std::move(plan.lock));owner->read_pages();return owner;});
    }
    const char* last_logit_setting = std::getenv("STRATA_DSV4_LAST_LOGIT_ONLY");
    if (last_logit_setting && std::strcmp(last_logit_setting, "0") != 0 &&
        std::strcmp(last_logit_setting, "1") != 0)
        throw std::runtime_error("last logit only must be0 or1");
    const bool last_logit_only = last_logit_setting && std::strcmp(last_logit_setting, "1") == 0;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = options_.n_ctx;
    context_params.n_batch = options_.n_batch;
    context_params.n_ubatch = options_.n_ubatch;
    context_params.n_seq_max = 1;
    glm_verify4_probe_=glm_mtp_&&strata::glm5_next::verify4_probe_requested();
    if(mtp_enabled()||glm_mtp_)context_params.n_rs_seq=2;
    if(glm_mtp_)context_params.n_outputs_max_per_seq=2;
    if(glm_verify4_probe_) {context_params.n_rs_seq=3;context_params.n_outputs_max_per_seq=4;}
    // This driver requests at most the final token's logits in any decode
    // batch. Bound worst-case head/output reservation to that same contract.
    if (last_logit_only) context_params.n_outputs_max = (mtp_enabled()||glm_mtp_) ? 2 : 1;
    if (last_logit_only&&glm_verify4_probe_) context_params.n_outputs_max=4;
    const int support_threads=model_->arch==LLM_ARCH_GLM5_NEXT&&strata::glm5_next::single_ccd_requested()?1:options_.n_threads;
    context_params.n_threads = support_threads;
    context_params.n_threads_batch = support_threads;
    // NextN metadata may remain; this context never builds or executes MTP.
    context_params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    const ggml_type kv_type = options_.kv_type == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    context_params.type_k = kv_type;
    context_params.type_v = kv_type;
    // DSV4 keeps distant history in compressed CSA/HCA/LID caches. Only its
    // raw SWA cache is ringed; the logical context remains options_.n_ctx.
    context_params.swa_full = options_.kv_resident == 0;
    context_params.dsv4_raw_kv_resident = options_.kv_resident;
    context_params.flash_attn_type = options_.flash_attn == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
        options_.flash_attn == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED :
        options_.flash_attn == "auto" ? LLAMA_FLASH_ATTN_TYPE_AUTO :
        model_->arch == LLM_ARCH_GLM5_NEXT ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.moe_cache_size = size_t(options_.moe_cache_mib) * 1024 * 1024;
    context_params.offload_kqv = true;
    context_params.op_offload = true;
    context_params.no_perf = false;
    context_params.cb_eval = callback;
    context_params.cb_eval_user_data = data;
    runtime_memory("before-context-init");
    context_ = llama_init_from_model(model_, context_params);
    runtime_memory("after-context-init");
    if (!context_) {
        error = "canonical DeepSeek4 context initialization failed";
        return false;
    }
    if(glm_mtp_)context_->set_embeddings_nextn(true,false);
    if(mtp_enabled()){std::cerr<<"strata_mtp_stage=context_reserved\n"<<std::flush;llama_synchronize(context_);}
    // GLM's full CCD belongs to the expert service. Small context/NextN CPU
    // operations share the submitting thread instead of another spinning team.
    auto pool_params = ggml_threadpool_params_default(support_threads);
    if(model_->arch==LLM_ARCH_GLM5_NEXT&&strata::glm5_next::single_ccd_requested())pool_params.poll=0;
    threadpool_ = ggml_threadpool_new(&pool_params);
    if (!threadpool_) {
        error = "canonical resident CPU threadpool initialization failed";
        return false;
    }
    llama_attach_threadpool(context_, threadpool_, threadpool_);
    if (options_.cache_leaf_aliases) configure_cache_leaf_aliases(context_);
    if (options_.pinned_experts) configure_expert_staging(context_, true, &retained_source_pin_,&retained_packed_);
    const auto pin=source_pin_metrics();
    std::fprintf(stderr,"strata_source_pin requested=%llu registered=%llu attempts=%llu failures=%llu last_error=%u register_ms=%.3f\n",
        (unsigned long long)expert_source_pin_budget(),(unsigned long long)pin.registered_bytes,
        (unsigned long long)pin.attempts,(unsigned long long)pin.failures,pin.last_error,pin.registration_wall_ns/1e6);
    runtime_memory("after-source-provider");
    if(warming.valid()){
        page_warm_=warming.get();page_warm_->lock_pages();const auto& w=page_warm_->metrics();
        std::fprintf(stderr,"strata_expert_page_warm bytes=%llu pages=%llu read_ms=%.3f locked_bytes=%llu lock_failures=%llu lock_ms=%.3f working_set_adjusted=%d working_set_requested_bytes=%llu working_set_error=%llu checksum=%llu\n",
            (unsigned long long)w.bytes,(unsigned long long)w.pages,w.read_ms,
            (unsigned long long)w.locked_bytes,(unsigned long long)w.lock_failures,w.lock_ms,w.working_set_adjusted?1:0,
            (unsigned long long)w.working_set_requested_bytes,(unsigned long long)w.working_set_error,(unsigned long long)w.checksum);
    }
    runtime_memory("after-page-warm");
    if (options_.cold_threads) configure_cold_experts(context_, options_.cold_threads, options_.cold_admit_free != 0);
    if(mtp_enabled()){std::cerr<<"strata_mtp_stage=source_provider_ready\n"<<std::flush;llama_synchronize(context_);}
    if (device_expert_cache_requested()) configure_device_expert_cache(context_);
    runtime_memory("after-device-cache");
    if(mtp_enabled()){std::cerr<<"strata_mtp_stage=expert_cache_ready\n"<<std::flush;llama_synchronize(context_);}
    return true;
}

bool Runtime::reconfigure_context(const Options& options, std::string& error,
                                  ggml_backend_sched_eval_callback callback, void* data) {
    if (!validate_options(options, error)) return false;
    if (options.model != options_.model || options.device != options_.device ||
        options.gpu_layers != options_.gpu_layers || options.mmproj != options_.mmproj) {
        error = "context reuse cannot change model/device/dense placement/image source";
        return false;
    }
    glm_session_.reset();
    mtp_drafter_.reset();
    if (context_) {
        llama_detach_threadpool(context_);
        llama_free(context_);
        context_ = nullptr;
    }
    if (threadpool_) { ggml_threadpool_free(threadpool_); threadpool_ = nullptr; }
    options_ = options;
    position_ = 0;
    try { return initialize_context(error, callback, data); }
    catch (const std::exception& failure) { error = failure.what(); return false; }
}

Runtime::~Runtime() {
    glm_session_.reset();
    mtp_drafter_.reset();
    if (context_) {
        llama_detach_threadpool(context_);
        llama_free(context_);
    }
    // Context-owned CPU backends are gone before their borrowed pool is freed.
    if (threadpool_) ggml_threadpool_free(threadpool_);
    page_warm_.reset();
    retained_source_pin_.reset();
    retained_packed_.reset();
    if (model_) llama_model_free(model_);
    mtp_weights_.reset();
    if(mtp_loading_backend_)ggml_backend_free(mtp_loading_backend_);
    // Backend registry is process-global; another Runtime may still be alive.
}

std::vector<llama_token> Runtime::tokenize(std::string_view text, bool add_special, bool parse_special) const {
    if (text.size() > size_t(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("prompt exceeds tokenizer length limit");
    }
    const int32_t needed = llama_tokenize(vocab_, text.data(), int32_t(text.size()), nullptr, 0,
                                          add_special, parse_special);
    if (needed == std::numeric_limits<int32_t>::min()) throw std::length_error("token count overflow");
    if (needed == 0) return {};
    if (needed > 0) throw std::runtime_error("unexpected tokenizer sizing result");
    std::vector<llama_token> tokens(size_t(-needed));
    const int32_t count = llama_tokenize(vocab_, text.data(), int32_t(text.size()), tokens.data(),
                                        int32_t(tokens.size()), add_special, parse_special);
    if (count < 0) throw std::runtime_error("tokenization failed after sizing");
    tokens.resize(size_t(count));
    return tokens;
}

std::string Runtime::piece(llama_token token, bool special) const {
    std::string output(64, '\0');
    int32_t count = llama_token_to_piece(vocab_, token, output.data(), int32_t(output.size()), 0, special);
    if (count < 0) {
        if (count == std::numeric_limits<int32_t>::min()) throw std::length_error("token piece overflow");
        output.resize(size_t(-count));
        count = llama_token_to_piece(vocab_, token, output.data(), int32_t(output.size()), 0, special);
    }
    if (count < 0) throw std::runtime_error("token detokenization failed");
    output.resize(size_t(count));
    return output;
}

std::string Runtime::detokenize(const std::vector<llama_token>& tokens, bool special) const {
    if (tokens.size() > size_t(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("detokenization token count overflows");
    }
    const int32_t needed = llama_detokenize(vocab_, tokens.data(), int32_t(tokens.size()),
                                            nullptr, 0, !special, special);
    if (needed == 0) return {};
    if (needed == std::numeric_limits<int32_t>::min()) throw std::length_error("detokenization size overflow");
    if (needed > 0) throw std::runtime_error("unexpected detokenization sizing result");
    std::string result(size_t(-needed), '\0');
    const int32_t count = llama_detokenize(vocab_, tokens.data(), int32_t(tokens.size()), result.data(),
                                           int32_t(result.size()), !special, special);
    if (count < 0) throw std::runtime_error("detokenization failed after sizing");
    result.resize(size_t(count));
    return result;
}

llama_token Runtime::greedy(bool ignore_eog) const {
    return greedy_at(-1,ignore_eog);
}
llama_token Runtime::greedy_at(int output_index,bool ignore_eog) const {
    const float* logits = llama_get_logits_ith(context_, output_index);
    if (!logits) throw std::runtime_error("no output logits are available");
    const int32_t count = llama_vocab_n_tokens(vocab_);
    if (count <= 0) throw std::runtime_error("model vocabulary is empty");
    llama_token best = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t index = 0; index < count; ++index) {
        if (std::isnan(logits[index])) throw std::runtime_error("non-finite output logits");
        if (logits[index] > maximum && (!ignore_eog || !llama_vocab_is_eog(vocab_, index))) {
            maximum = logits[index];
            best = index;
        }
    }
    if (!std::isfinite(maximum)) throw std::runtime_error("no finite output logits");
    idle_device_expert_cache(context_);
    return best;
}

bool Runtime::is_eog(llama_token token) const noexcept { return llama_vocab_is_eog(vocab_, token); }

void Runtime::reset(bool clear_data) {
    glm_catchup_pending_=false;
    if(glm_session_)glm_session_->reset();
    glm_prefix_hidden_.clear();glm_latest_hidden_.clear();glm_collect_prefix_=true;
    if(mtp_drafter_)mtp_drafter_->reset();
    if(mtp_enabled())context_->strata_capture_nextn_tail(128);
    if(model_->arch==LLM_ARCH_GLM5_NEXT && pp_stream_enabled(true))context_->strata_moe_cache()->strata_clear_glm_prefill();
    llama_memory_clear(llama_get_memory(context_), clear_data);
    llama_perf_context_reset(context_);
    position_ = 0;
}

llama_perf_context_data Runtime::performance() const { return llama_perf_context(context_); }

ExpertStagingMetrics Runtime::expert_transport_metrics() const { return expert_staging_metrics(context_); }
SourcePinMetrics Runtime::source_pin_metrics() const { return expert_source_pin_metrics(context_); }
ColdExpertMetrics Runtime::cold_metrics() const { return cold_expert_metrics(context_); }

std::string Runtime::metadata(const char* key) const {
    const int32_t length = llama_model_meta_val_str(model_, key, nullptr, 0);
    if (length < 0) return {};
    std::string value(size_t(length) + 1, '\0');
    if (llama_model_meta_val_str(model_, key, value.data(), value.size()) < 0) return {};
    value.resize(size_t(length));
    return value;
}

const char* Runtime::source_commit() noexcept { return STRATA_DEEPSEEK_LLAMA_COMMIT; }

bool Runtime::image_api_available() noexcept {
#ifdef STRATA_DEEPSEEK_WITH_MTMD
    return true;
#else
    return false;
#endif
}

}  // namespace strata::deepseek_v4
