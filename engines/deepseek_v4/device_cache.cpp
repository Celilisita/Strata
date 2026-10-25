#include "device_cache.hpp"
#include "../glm5_next/partial_down_host.hpp"
#include "../glm5_next/partial_down_graph.hpp"
#include "../glm5_next/hybrid_provider.hpp"
#include "../glm5_next/route_capture.hpp"
#include "../glm5_next/phase_options.hpp"
#include "../glm5_next/mtp_options.hpp"
#include "../glm5_next/background_options.hpp"
#include "../glm5_next/hybrid_overlap.hpp"
#include "../glm5_next/verify4_options.hpp"
#include "../glm5_next/cold_gpu_owner.hpp"
#include "../glm5_next/cold_gpu_protocol.hpp"
#include "../glm5_next/cpu_gu_q8_policy.hpp"
#include "cuda_pool_retirement.hpp"
#include "device_cache_budget.hpp"
#include "cache_growth.hpp"
#include <cstdio>
#include <chrono>
#include "expert_frequency.hpp"
#include "native_tg_graph.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
bool glm_dma_option(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !std::strcmp(value, "0")) return false;
    if (!std::strcmp(value, "1")) return true;
    throw std::runtime_error(std::string(name) + " must be 0 or 1");
}
bool pp_worker_parking_enabled(){
    const char* value=std::getenv("STRATA_DSV4_PP_WORKER_PARK");
    if(!value)return true;
    if(!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::runtime_error("PP worker parking must be0 or1");
}
}
#include "../glm5_next/phase_capacity_service.inc"
#include "../glm5_next/phase_service.inc"
bool DeviceExpertCache::phase_change(bool single_token) {
    if(model_.arch==LLM_ARCH_GLM5_NEXT && strata::glm5_next::phase_cache_requested()){
        const bool due=(enabled_&&single_token)!=glm_tg_phase_||glm_capacity_due(single_token);
        if(due&&borrowed_retire_)borrowed_retire_(borrowed_retire_user_);return due;
    }
    return enabled_ && single_token!=active_;
}
bool device_expert_cache_requested() {
    const char* value = std::getenv("STRATA_DSV4_DEVICE_CACHE");
    if (value && std::strcmp(value, "0") != 0 && std::strcmp(value, "1") != 0)
        throw std::runtime_error("device cache must be0 or1");
    return value && std::strcmp(value, "1") == 0;
}
void configure_device_cache_process() {
    if (!device_expert_cache_requested()) return;
    // Set before CUDA initialization; lazy loads can wait for a running miss lookup.
#ifdef _WIN32
    const int code = _putenv_s("CUDA_MODULE_LOADING", "EAGER") | _putenv_s("CUDA_MODULE_DATA_LOADING", "EAGER");
#else
    const int code = setenv("CUDA_MODULE_LOADING", "EAGER", 1) | setenv("CUDA_MODULE_DATA_LOADING", "EAGER", 1);
#endif
    if (code) throw std::runtime_error("cannot configure CUDA eager module loading");
}
DeviceExpertCache::DeviceExpertCache(const llama_model& model, llama_moe_cache& cache,
                                   ggml_backend_sched_t scheduler, ExpertStaging* staging) :
    model_(model), staging_(staging), cache_(cache), scheduler_(scheduler),
    causal_host_(model.arch==LLM_ARCH_GLM5_NEXT&&strata::glm5_next::causal_phases_requested()) {
    const auto stage=[](const char* name){if(std::getenv("STRATA_GLM53_CRASH_TRACE")){std::fprintf(stderr,"strata_device_init %s\n",name);std::fflush(stderr);}};stage("begin");
    const bool glm = model.arch == LLM_ARCH_GLM5_NEXT;
    if ((!glm && (model.arch != LLM_ARCH_DEEPSEEK4 || (model.layers.size() != 43 && model.layers.size() != 44))) ||
        model.layers.size() > device_cache_max_layers || !scheduler)
        throw std::runtime_error("device cache model identity differs");
    experts_ = int(model.hparams.n_expert); top_k_ = int(model.hparams.n_expert_used_max());
    if ((experts_ != 256 && experts_ != 288) || (top_k_ != 6 && top_k_ != 8) || (glm && (native_tg_enabled() || native_tg_cpu_only() || router_prefetch_enabled())))
        throw std::runtime_error("device lookup requires a supported layout; GLM native compute remains separately gated");
    static std::atomic<uint64_t> epochs{0}; owner_epoch_ = ++epochs;
    directory_host_.assign(device_cache_max_layers * experts_, -1);
    pp_permanent_slots_=int(cache.strata_cost_snapshot().permanent_slots);
    frequency_ = expert_frequency_enabled() ? cache.strata_frequency() : nullptr;
    if (expert_frequency_enabled() && !frequency_) throw std::runtime_error("canonical frequency authority is absent");
    const char* graph_opt = std::getenv("GGML_CUDA_GRAPH_OPT");
    if (graph_opt && std::atoi(graph_opt) == 1)
        throw std::runtime_error("one mapped miss packet requires sequential CUDA lookup scheduling");
    backend_ = cache.backend(glm ? int(model.hparams.n_layer_dense_lead) : 0);
    if (!backend_) throw std::runtime_error("device cache has no selected backend");
    try {
        stage("metadata");
        const bool cold_alias=glm&&staging_&&expert_source_alias_requested();
        context_ = ggml_init({size_t(49+cold_alias) * ggml_tensor_overhead(), nullptr, true});
        if (!context_) throw std::runtime_error("device cache metadata allocation failed");
        directory_ = ggml_new_tensor_1d(context_, GGML_TYPE_I32, device_cache_max_layers * experts_);
        status_ = ggml_new_tensor_1d(context_, GGML_TYPE_I32, 1);
        counters_ = ggml_new_tensor_1d(context_, GGML_TYPE_I64, device_cache_counter_count);
        frequency_host_.resize(model.layers.size()*experts_+1);
        if (frequency_) frequency_gpu_ = ggml_new_tensor_1d(context_, GGML_TYPE_I32, frequency_host_.size());
        if(glm&&strata::glm5_next::hybrid_overlap_requested())hybrid_ticket_=ggml_new_tensor_1d(context_,GGML_TYPE_I32,2);
        if(glm&&strata::glm5_next::RouteCapture::requested())
            route_capture_=std::make_unique<strata::glm5_next::RouteCapture>(context_);
        if(cold_alias)cold_gpu_=std::make_unique<strata::glm5_next::ColdGpuOwner>(model_,cache_,*staging_,context_,owner_epoch_);
        for (int layer = 0; layer < int(model_.layers.size()); ++layer) {
            if (glm && !model.layers[layer].ffn_up_exps) continue;
            originals_[layer] = cache.strata_directory(layer);
            const auto* original = originals_[layer];
            const auto* bank = cache.get_experts(model.layers[layer].ffn_up_exps);
            if (cache.backend(layer) != backend_ || !original || !bank || !bank->data ||
                original->type != GGML_TYPE_I32 || ggml_nelements(original) != experts_ ||
                !ggml_is_contiguous(original) || !original->data || !original->buffer ||
                !ggml_backend_buffer_is_host(original->buffer) || bank->ne[2] < top_k_)
                throw std::runtime_error("device cache directory/bank contract differs at layer="+std::to_string(layer)+
                    " cached_rows="+std::to_string(bank?bank->ne[2]:-1)+" required="+std::to_string(top_k_));
            maps_[layer] = ggml_view_2d(context_, directory_, 1, experts_, sizeof(int32_t), layer * experts_ * sizeof(int32_t));
            ggml_format_name(maps_[layer], "strata_device_cache.map-%d", layer);
        }
        stage("buffer");
        buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend_);
        if (!buffer_) throw std::runtime_error("device cache directory allocation failed");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if(cold_gpu_)cold_gpu_->initialize();
        const int32_t empty = 0;
        const std::array<uint64_t, device_cache_counter_count> counters{};
        ggml_backend_tensor_set(status_, &empty, 0, sizeof(empty));
        ggml_backend_tensor_set(counters_, counters.data(), 0, sizeof(counters));
        if(hybrid_ticket_){const uint32_t zeros[2]{};ggml_backend_tensor_set(hybrid_ticket_,zeros,0,sizeof(zeros));}
        if(route_capture_)route_capture_->initialize();
        if (frequency_) {
            frequency_->export_state(frequency_host_.data(), frequency_host_.size());
            ggml_backend_tensor_set(frequency_gpu_, frequency_host_.data(), 0, frequency_host_.size()*sizeof(uint32_t));
        }
        stage("dma");
        // Existing blocking CUDA events and slab packing remain GLM-only opt-ins.
        dma_ = std::make_unique<DeviceCacheDma>(backend_, staging, true,
            glm && glm_dma_option("STRATA_GLM53_CUDA_BLOCKING_WAIT"),
            glm && glm_dma_option("STRATA_GLM53_DMA_PACK_FRAGMENTS"),
            glm && strata::glm5_next::causal_phases_requested());
        dma_->packet()->owner_epoch = owner_epoch_;
        if(hybrid_ticket_) {
            const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
            native_ready_=reinterpret_cast<decltype(native_ready_)>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_ready"));
            native_reply_=reinterpret_cast<decltype(native_reply_)>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_reply"));
            if(!native_ready_||!native_reply_)throw std::runtime_error("GLM overlap requires D4 system-atomic ticket publication");
        }
        const int headroom_slots=pp_cache_slots_with_headroom(pp_permanent_slots_);
        if(!glm && headroom_slots!=pp_permanent_slots_){cache_.strata_resize_phase(headroom_slots,upload,this);pp_permanent_slots_=headroom_slots;}
        trace_device_memory_budget(backend_,"cache_initialized",pp_permanent_slots_);
        if (native_tg_enabled()) {
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
            native_ready_ = reinterpret_cast<decltype(native_ready_)>(ggml_backend_reg_get_proc_address(reg, "strata_native_tg_ready"));
            native_reply_ = reinterpret_cast<decltype(native_reply_)>(ggml_backend_reg_get_proc_address(reg, "strata_native_tg_reply"));
            native_complete_ = reinterpret_cast<decltype(native_complete_)>(ggml_backend_reg_get_proc_address(reg, "strata_native_tg_complete"));
            using Reserve = bool(*)(ggml_backend_t);
            const auto reserve = reinterpret_cast<Reserve>(ggml_backend_reg_get_proc_address(reg, "strata_native_tg_reserve"));
            if (!native_ready_ || !native_reply_ || !native_complete_ || !reserve || !reserve(backend_))
                throw std::runtime_error("native TG backend service is absent");
            using Slots = uintptr_t(*)(ggml_backend_t);
            const auto slots = reinterpret_cast<Slots>(ggml_backend_reg_get_proc_address(reg, "strata_native_tg_slots"));
            native_slots_ = slots ? slots(backend_) : 0;
            if (!native_slots_) throw std::runtime_error("native TG preallocated slot storage is absent");
            const char* threads = std::getenv("STRATA_DSV4_NATIVE_CPU_THREADS");
            cpu_ = std::make_unique<Mxfp4CpuService>(threads ? std::atoi(threads) : 4);
            cpu_->set_spin_active(!pp_worker_parking_enabled());
        }
        if(router_prefetch_enabled())read_ahead_=std::make_unique<NativeReadAhead>(backend_,model_,*cpu_,dma_->packet(),native_ready_,stop_,worker_active_);
        dma_->set_active(false);
        stage("directory");
        project_directory();
        stage("worker");
        worker_ = std::thread(&DeviceExpertCache::worker, this);
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_.wait(lock, [&] { return worker_ready_; });
        if (!error_.empty()) throw std::runtime_error(error_);
    } catch (...) { release(); throw; }
}
DeviceExpertCache::~DeviceExpertCache() { release(); }
bool DeviceExpertCache::refresh_cold_gpu_rows() {
    bool cold_change=false,requested=false;
    if(cold_gpu_){
        requested=strata::glm5_next::ColdGpuOwner::requested();
        const auto maximum=strata::glm5_next::ColdGpuOwner::requested_max_rows();
        const auto topology=strata::glm5_next::ColdGpuOwner::topology_requested();
        const bool conditional4=strata::glm5_next::ColdGpuOwner::conditional4_requested();
        cold_change=requested!=cold_gpu_->requested_value()||maximum!=cold_gpu_->requested_max_rows_value()||
            topology!=cold_gpu_->topology_requested_value()||conditional4!=cold_gpu_->conditional4_requested_value();
    }
    const bool cpu_active=hybrid_&&hybrid_->gu_q8_reuse_requested()&&strata::glm5_next::requested_cpu_gu_q8_active();
    const bool cpu_change=hybrid_&&cpu_active!=hybrid_->gu_q8_reuse_active();
    if(!cold_change&&!cpu_change)return false;
    // This drains mapped-weight compute readers as well as ordinary kernels.
    ggml_backend_synchronize(backend_);
    std::lock_guard<std::mutex> lock(service_mutex_);
    finish_background(true);
    if(cpu_change&&!hybrid_->set_gu_q8_reuse_active(cpu_active))
        throw std::runtime_error("GU Q8 active request lacks construction-time capability");
    if(cold_change)cold_gpu_->refresh(requested,enabled_&&glm_tg_phase_);
    // CPU-only selection changes no GPU node, pointer, weight or capture shape.
    return cold_change; // Existing cold-row changes still invalidate GPU graphs.
}
uint32_t DeviceExpertCache::cold_gpu_mask(int32_t layer,int routes,uint32_t missing,const int32_t* directory) {
    return cold_gpu_?cold_gpu_->validate_ticket(layer,routes,missing,directory,*dma_->packet()):0;
}
void DeviceExpertCache::release() noexcept {
    if (backend_) ggml_backend_synchronize(backend_);
    stop_.store(true, std::memory_order_release);
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    read_ahead_.reset();cpu_.reset();hybrid_.reset();hybrid_four_scratch_.reset();route_capture_.reset();
    // Drain the copy stream before borrowed source registrations and banks die.
    if(cold_gpu_&&cold_gpu_->partial_down_phase(24))cold_gpu_->release_rows();
    dma_.reset();
    if(cold_gpu_)cold_gpu_->release_rows();
    cold_gpu_.reset();
    if (buffer_) { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
    if (context_) { ggml_free(context_); context_ = nullptr; }
}
void DeviceExpertCache::set_enabled(bool enabled) {
    ggml_backend_synchronize(backend_);
    std::lock_guard<std::mutex> lock(service_mutex_);
    finish_background(true);
    if(cold_gpu_)cold_gpu_->release_rows();
    if (active_ && frequency_) read_frequency();
    enabled_ = enabled;
    if(cold_gpu_&&enabled_&&glm_tg_phase_)cold_gpu_->refresh(strata::glm5_next::ColdGpuOwner::requested(),true);
    active_ = false;
    worker_active_.store(false, std::memory_order_release);
    dma_->set_active(false);
}
void DeviceExpertCache::prepare_partial_down_diagnostic(uint64_t generation){
    ggml_backend_synchronize(backend_);std::lock_guard<std::mutex> lock(service_mutex_);finish_background(true);
    if(!generation||!cold_gpu_||!hybrid_||!hybrid_active_||!glm_tg_phase_)
        throw std::runtime_error("partial Down diagnostic requires prepared hybrid TG");
    hybrid_->partial_down_diagnostic_service().prepare_partial_down_diagnostic(24,10,6);
    dma_->prepare_partial_down_diagnostic();
    cold_gpu_->prepare_partial_down_diagnostic(backend_,dma_->partial_down_phase(),dma_->partial_down_device_phase(),generation);
}
void DeviceExpertCache::prepare(bool single_token) {
    if(glm_capacity_poisoned_)throw std::runtime_error("GLM capacity owner poisoned");
    hybrid_active_=enabled_&&model_.arch==LLM_ARCH_GLM5_NEXT&&strata::glm5_next::hybrid_requested();
    if(hybrid_active_&&!frequency_)throw std::runtime_error("hybrid requires the shared frequency authority");
    if(hybrid_active_&&!hybrid_){
        const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
        const auto ready=reinterpret_cast<bool(*)()>(ggml_backend_reg_get_proc_address(reg,"strata_glm_hybrid_reduce_ready"));
        if(!ready||!ready())throw std::runtime_error("GLM fused contribution kernel was not compiled into this backend");
        const char* curve=std::getenv("STRATA_GLM53_PROVIDER_CURVES");
        if(!curve)throw std::runtime_error("GLM hybrid lacks calibrated provider curves");
        const bool four=strata::glm5_next::verify4_probe_requested();
        if(four&&(!hybrid_ticket_||route_capture_))
            throw std::runtime_error("four-position diagnostic requires overlap and disabled route capture");
        hybrid_=std::make_unique<strata::glm5_next::HybridProvider>(model_,curve,ggml_backend_dev_description(ggml_backend_get_device(backend_)),four);
        if(four)hybrid_four_scratch_=std::make_unique<strata::glm5_next::HybridFourScratch>();
    }
    const bool next = enabled_ && single_token;
    if(active_&&!next){
        ggml_backend_synchronize(backend_);
        worker_active_.store(false,std::memory_order_release);
        {std::lock_guard<std::mutex> lock(service_mutex_);
            if(cpu_){cpu_->wait();cpu_->set_spin_active(!pp_worker_parking_enabled());}}
    }
    if (active_ && !next && frequency_) {
        std::lock_guard<std::mutex> lock(service_mutex_);
        finish_background(true);
        read_frequency();
        if (!(model_.arch==LLM_ARCH_GLM5_NEXT && strata::glm5_next::phase_cache_requested()) && pp_permanent_slots_ && cache_.strata_cost_snapshot().permanent_slots != uint64_t(pp_permanent_slots_)) {
            cache_.strata_resize_phase(pp_permanent_slots_,upload,this);
            project_directory();
        }
    }
    if(!next && glm_tg_phase_){std::lock_guard<std::mutex> lock(service_mutex_);glm_phase_transition(false);}
    if (next && !active_) {
        ggml_backend_synchronize(backend_);
        if(cpu_)cpu_->set_spin_active(true);
        std::lock_guard<std::mutex> lock(service_mutex_);
        if (!error_.empty()) throw std::runtime_error(error_);
        using EnterTg = bool (*)(ggml_backend_t);
        const auto registration = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
        const auto retire = reinterpret_cast<EnterTg>(ggml_backend_reg_get_proc_address(registration,"strata_pp_stream_enter_tg"));
        if (!retire || !retire(backend_)) throw std::runtime_error("PP to TG scratch retirement unavailable");
        const char* pool_retirement=std::getenv("STRATA_DSV4_RETIRE_CUDA_POOL");
        if(pool_retirement&&!std::strcmp(pool_retirement,"1")) {
            auto inspect=reinterpret_cast<CudaPoolInspect>(ggml_backend_reg_get_proc_address(registration,"strata_cuda_pool_inspect"));
            auto retire_pool=reinterpret_cast<CudaPoolRetire>(ggml_backend_reg_get_proc_address(registration,"strata_cuda_pool_retire"));CudaPoolRetirement state;
            if(!inspect||!retire_pool||!inspect(backend_,&state))throw std::runtime_error("CUDA pool still holds live temporary allocations at TG boundary");
            if(state.retained_bytes&& !retire_pool(backend_,&state))
                throw std::runtime_error("CUDA pool retirement refused a non-quiescent owner");
            std::fprintf(stderr,"strata_cuda_pool retained_bytes=%llu released_bytes=%llu graphs_destroyed=%llu\n",
                (unsigned long long)state.retained_bytes,(unsigned long long)state.released_bytes,(unsigned long long)state.graphs_destroyed);
        }
        const char* grow=std::getenv("STRATA_DSV4_TG_CACHE_GROW_MIB");
        if(native_tg_cpu_only()&&grow&&std::atoi(grow)>0) {
            const char* headroom=std::getenv("STRATA_DSV4_TG_HEADROOM_MIB");
            const int margin_mib=headroom?std::atoi(headroom):1024;
            if(margin_mib<512||margin_mib>2048)throw std::runtime_error("TG WDDM headroom must be512..2048 MiB");
            const auto budget=device_memory_budget(backend_);
            if(margin_mib<1024&&!budget.wddm_available)
                throw std::runtime_error("Reduced TG headroom requires live WDDM budget telemetry");
            const size_t margin=size_t(margin_mib)<<20,requested=size_t(std::atoi(grow))<<20;
            const size_t safety=128ull<<20,threshold=margin+safety;
            const size_t wddm_free=budget.wddm_available&&budget.local_budget>budget.local_usage?
                size_t(budget.local_budget-budget.local_usage):budget.cuda_free;
            const size_t free=std::min(size_t(budget.cuda_free),wddm_free);
            const size_t available=free>threshold?std::min(requested,free-threshold):0;
            const int maximum=int(available/13369344);
            const auto growth=converge_cache_growth(maximum,
                [&](int slots){cache_.strata_resize_phase(pp_permanent_slots_+slots,upload,this);},
                [&]{
                    const auto measured=device_memory_budget(backend_);
                    const size_t local_free=measured.wddm_available&&measured.local_budget>measured.local_usage?
                        size_t(measured.local_budget-measured.local_usage):measured.cuda_free;
                    return measured.cuda_free>=threshold&&local_free>=threshold;
                });
            const auto after=device_memory_budget(backend_);
            std::fprintf(stderr,"strata_tg_cache_growth added_slots=%d attempts=%d rejected=%d margin_mib=%d safety_mib=128 cuda_free_before=%llu wddm_free_before=%llu cuda_free_after=%llu wddm_free_after=%llu\n",
                growth.added_slots,growth.attempts,growth.rejected,margin_mib,
                (unsigned long long)budget.cuda_free,(unsigned long long)wddm_free,
                (unsigned long long)after.cuda_free,
                (unsigned long long)(after.wddm_available&&after.local_budget>after.local_usage?
                    after.local_budget-after.local_usage:after.cuda_free));
            cache_.strata_use_prefill_frequency();
            cache_.strata_prewarm_phase(upload,this);
        }
        glm_phase_transition(true);
        if (frequency_) {
            frequency_->export_state(frequency_host_.data(), frequency_host_.size());
            ggml_backend_tensor_set(frequency_gpu_, frequency_host_.data(), 0, frequency_host_.size()*sizeof(uint32_t));
        }
        if(route_capture_)route_capture_->seed(model_,cache_,frequency_host_);
        if(!(model_.arch==LLM_ARCH_GLM5_NEXT && strata::glm5_next::phase_cache_requested()))project_directory();
    }
    if(next&&active_&&glm_capacity_pending_)glm_capacity_recheck();
    active_ = next;
    if(next!=worker_active_.load(std::memory_order_acquire))
        trace_device_memory_budget(backend_,next?"tg_ready":"pp_ready",int(cache_.strata_cost_snapshot().permanent_slots));
    dma_->set_active(next);
    worker_active_.store(next, std::memory_order_release);
    if (next) wake_.notify_one();
}
void DeviceExpertCache::verify() const {
    int32_t status = -1;
    ggml_backend_tensor_get(status_, &status, 0, sizeof(status));
    std::lock_guard<std::mutex> lock(service_mutex_);
    if (status || !error_.empty())
        throw std::runtime_error(error_.empty() ? "device cache miss failed or timed out: layer=" + std::to_string(dma_->packet()->layer) +
            " sequence=" + std::to_string(dma_->packet()->sequence) + " response=" + std::to_string(dma_->packet()->response) + " service_ns=" + std::to_string(service_ns_) +
            " memcpy_ns=" + std::to_string(dma_->metrics().host_copy_wall_ns) +
            " slab_wait_ns=" + std::to_string(dma_->metrics().slab_wait_wall_ns) +
            " submit_ns=" + std::to_string(dma_->costs().submit_ns) +
            " commit_wait_ns=" + std::to_string(dma_->costs().commit_wait_ns) +
            " planning_ns=" + std::to_string(planning_ns_) +
            " projection_ns=" + std::to_string(projection_ns_) : error_);
}
DeviceCacheMetrics DeviceExpertCache::metrics() const {
    DeviceCacheMetrics result;
    ggml_backend_tensor_get(counters_, result.gpu.data(), 0, sizeof(result.gpu));
    std::lock_guard<std::mutex> lock(service_mutex_);
    result.activation_control_d2h_bytes=activation_control_bytes_;
    result.cpu_contribution_h2d_bytes=contribution_bytes_;result.hybrid_gpu_services=hybrid_gpu_services_;
    result.last_miss_ids = last_miss_ids_;
    result.last_miss_counts = last_miss_counts_;
    result.host_services = services_;
    result.service_wall_ns = service_ns_;
    result.service_max_ns = service_max_ns_;
    result.directory_h2d_bytes = directory_bytes_;
    result.transport = dma_->metrics();
    result.dma_costs = dma_->costs();
    result.last_phase_budget=last_phase_budget_;result.glm_capacity=glm_capacity_;
    result.phase_transitions=phase_transitions_;result.phase_resize_ns=phase_resize_ns_;
    result.phase_seed_ns=phase_seed_ns_;result.phase_seed_bytes=phase_seed_bytes_;result.phase_verify_bytes=phase_verify_bytes_;result.phase_evicted_keys=phase_evicted_keys_;
    result.planning_ns = planning_ns_;
    result.projection_ns = projection_ns_;
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
    using Read = bool (*)(ggml_backend_t, PpStreamMetrics*);
    const auto read = reinterpret_cast<Read>(ggml_backend_reg_get_proc_address(reg, "strata_pp_stream_metrics"));
    if (read) read(backend_, &result.pp);
    if(pp_receipt_available_)result.pp_plan_receipt=read_pp_plan_receipt();
    result.capacity = cache_.strata_cost_snapshot();
    if(model_.arch==LLM_ARCH_GLM5_NEXT)result.prefill_coverage=cache_.strata_glm_prefill_coverage();
    trace_device_memory_budget(backend_,active_?"tg_snapshot":"pp_snapshot",int(result.capacity.permanent_slots));
    if(read_ahead_)result.read_ahead=read_ahead_->metrics();
    result.background_verify_bytes=background_verify_bytes_;result.background_layout_commits=background_layout_commits_;
    result.background_bytes=background_bytes_;result.background_commits=background_commits_;
    result.background_staged_bytes=background_staged_bytes_;result.background_staged_experts=background_staged_experts_;
    result.cpu_jobs = cpu_jobs_.load(); result.cpu_routes = cpu_routes_.load();
    result.hybrid_overlapped_services=hybrid_overlapped_services_;
    if(cold_gpu_){result.cold_gpu=cold_gpu_->metrics();result.cold_gpu_reserved_rows=cold_gpu_->reserved_rows();result.cold_gpu_metadata_bytes=cold_gpu_->metadata_bytes();
        const auto jobs=cold_gpu_->selected_job_counts();result.cold_gpu_two_jobs=jobs[0];result.cold_gpu_four_jobs=jobs[1];
        result.cold_gpu_requested_max_rows=cold_gpu_->requested_max_rows_value();}
    if(cold_gpu_){result.cold_gpu_topology=cold_gpu_->topology_metrics();
        result.cold_gpu_topology_requested=cold_gpu_->topology_requested_value();
        result.cold_gpu_repeat_priority=cold_gpu_->repeat_priority_metrics();
        result.cold_gpu_conditional4=cold_gpu_->conditional4_metrics();
        result.cold_gpu_conditional4_requested=cold_gpu_->conditional4_requested_value();
        result.cold_gpu_conditional4_active=cold_gpu_->conditional4_active();}
    result.causal_host=causal_host_.snapshot(owner_epoch_);
    result.cpu_logical_weight_bytes = cpu_bytes_.load(); result.cpu_compute_ns = cpu_ns_.load();
    if(hybrid_){result.cpu_service_threads=hybrid_->threads();result.cpu_pair_jobs=hybrid_->pair_jobs();result.cpu_pair_ns=hybrid_->pair_ns();}
    if(hybrid_)result.cpu_gu_q8_preparation=hybrid_->gu_q8_preparation();
    if(hybrid_four_scratch_) {
        result.cpu_four_jobs=hybrid_->four_jobs();result.cpu_four_ns=hybrid_->four_ns();
        const auto& a=hybrid_->four_capacity_before();const auto& b=hybrid_->four_capacity_after();
        result.cpu_four_capacity_before={a.prepared_graphs,a.scheduler_cpu_buffer_bytes,a.activation_capacity_bytes};
        result.cpu_four_capacity_after={b.prepared_graphs,b.scheduler_cpu_buffer_bytes,b.activation_capacity_bytes};
        result.cpu_four_prepare_ms=hybrid_->four_prepare_ms();result.cpu_four_scratch_bytes=sizeof(*hybrid_four_scratch_);
    }
    return result;
}
ggml_tensor* DeviceExpertCache::slot_map(int32_t layer, int64_t tokens, int64_t used,
                                       ggml_tensor* original, void* data) {
    auto& self = *static_cast<DeviceExpertCache*>(data);
    self.graph_active_ = self.active_ && (tokens == 1 || (tokens == 2 && (native_tg_cpu_only() || (self.model_.arch==LLM_ARCH_GLM5_NEXT && strata::glm5_next::mtp_requested()))));
    if(tokens==4&&self.model_.arch==LLM_ARCH_GLM5_NEXT&&self.hybrid_four_scratch_) {
        if(!self.hybrid_active_||!self.hybrid_->four_positions_prepared())
            throw std::runtime_error("four-position lookup has no prepared CPU provider");
        self.graph_active_=self.active_;
    }
    if (!self.graph_active_) return original;
    if(self.model_.arch==LLM_ARCH_GLM5_NEXT&&tokens==2&&!self.hybrid_active_)throw std::runtime_error("GLM verification requires validated hybrid pair service");
    if (layer < 0 || layer >= int(self.model_.layers.size()) || used != self.top_k_ || original != self.originals_[layer])
        throw std::runtime_error("device cache lookup shape differs");
    return self.maps_[layer];
}
void DeviceExpertCache::decorate(ggml_tensor* tensor, int32_t layer, void* data) {
    auto& self = *static_cast<DeviceExpertCache*>(data);
    if (!self.graph_active_) return;
    if (layer < 0 || layer >= int(self.model_.layers.size()) || tensor->op != GGML_OP_GET_ROWS || tensor->src[0] != self.maps_[layer] ||
        !tensor->src[1] || (ggml_nelements(tensor->src[1]) != self.top_k_ && ggml_nelements(tensor->src[1]) != 2*self.top_k_&&
        !(self.hybrid_four_scratch_&&ggml_nelements(tensor->src[1])==4*self.top_k_)) || !ggml_is_contiguous(tensor->src[1]))
        throw std::runtime_error("device cache GET_ROWS seam differs");
    tensor->src[2] = self.status_;
    tensor->src[3] = self.counters_;
    tensor->src[4] = self.frequency_gpu_;
    tensor->src[6] = self.route_capture_?self.route_capture_->tensor():nullptr;
    tensor->src[7] = self.hybrid_ticket_;
    const uintptr_t packet = self.dma_->device_packet();
    const uint64_t timeout = self.dma_->timeout_ns();
    static_assert(sizeof(packet) <= 2 * sizeof(int32_t));
    std::memcpy(tensor->op_params, &packet, sizeof(packet));
    tensor->op_params[2] = layer;
    tensor->op_params[3] = self.experts_;
    tensor->op_params[10] = self.hybrid_active_?1:0;
    tensor->op_params[11] = self.hybrid_active_&&self.hybrid_ticket_&&self.hybrid_->cpu_only_layer(layer)?1:0;
    // No hook means the original GPU template: no extra launch or packet write.
    const uintptr_t cold=self.cold_gpu_&&ggml_nelements(tensor->src[1])==16&&tensor->op_params[11]?
        self.cold_gpu_->device_layer(layer):0;
    std::memcpy(tensor->op_params+strata::glm5_next::cold_gpu_pointer_word,&cold,sizeof(cold));
    tensor->op_params[strata::glm5_next::cold_gpu_magic_word]=cold?
        (self.cold_gpu_->reserved_rows()==4?strata::glm5_next::cold_gpu_op_magic4:strata::glm5_next::cold_gpu_op_magic):0;
    if(ggml_nelements(tensor->src[1])==32) {
        if(!self.hybrid_active_||!self.hybrid_ticket_||!self.hybrid_four_scratch_||self.route_capture_)
            throw std::runtime_error("four-position lookup must use bounded CPU overlap diagnostic");
        tensor->op_params[11]=1;
    }
    std::memcpy(tensor->op_params + 6, &self.owner_epoch_, sizeof(self.owner_epoch_));
    tensor->op_params[8] = int(self.model_.hparams.n_layer_dense_lead);
    tensor->op_params[9] = int(self.model_.hparams.n_layer()) - 1;
    std::memcpy(tensor->op_params + 4, &timeout, sizeof(timeout));
    tensor->op_params[device_cache_tag_index] = device_cache_lookup_tag;
    if(cold&&self.cold_gpu_->partial_down_phase(layer))
        strata::glm5_next::partial_down_attach_lookup(tensor,self.cold_gpu_->partial_down_binding(layer));
}
ggml_tensor* DeviceExpertCache::guard(ggml_context* context, ggml_tensor* logits, void* data) {
    auto& self = *static_cast<DeviceExpertCache*>(data);
    if (!self.graph_active_) return logits;
    if (!logits || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits))
        throw std::runtime_error("device cache logits seam differs");
    auto* result = ggml_clamp(context, logits, -INFINITY, INFINITY);
    result->src[1] = self.status_;
    result->op_params[device_cache_tag_index] = device_cache_guard_tag;
    ggml_set_name(result, "strata_device_cache.logits_guard");
    ggml_backend_sched_set_tensor_backend(self.scheduler_, result, self.backend_);
    return result;
}
}

namespace strata::deepseek_v4 {
#include "../glm5_next/background_service.inc"

}

namespace strata::deepseek_v4 {
void DeviceExpertCache::write_routing_seed(const std::string& path) const {
    if(model_.arch==LLM_ARCH_GLM5_NEXT)throw std::runtime_error("GLM routing seed export is not implemented");
    std::lock_guard<std::mutex> lock(service_mutex_);std::ofstream out(path);
    out<<"{\"permanent_slots\":"<<cache_.strata_cost_snapshot().permanent_slots<<",\"directory\":[";
    for(int key=0;key<int(model_.layers.size())*256;++key){if(key)out<<',';out<<static_cast<const int32_t*>(originals_[key/256]->data)[key%256];}
    out<<"],\"prefill_frequency\":[";
    for(size_t key=0;key<frequency_host_.size();++key){if(key)out<<',';out<<frequency_host_[key];}
    out<<"],\"background_dma_eligible\":[";
    const char* stage=std::getenv("STRATA_DSV4_BG_STAGE_UNPINNED");const bool staged=stage&&!std::strcmp(stage,"1");
    for(int key=0;key<int(model_.layers.size())*256;++key){bool ready=staging_!=nullptr;const auto& layer=model_.layers[key/256];
        for(auto* source:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps})if(staging_){const auto part=staging_->transport_segment(static_cast<const char*>(source->data)+(key%256)*source->nb[2],source->nb[2]);ready&=part.registered&&part.bytes>=source->nb[2];}
        if(staged&&key/256<43)ready=true;
        if(key)out<<',';out<<(ready?1:0);
    }
    out<<"]}\n";if(!out)throw std::runtime_error("prefill routing seed receipt failed");
}
}

namespace strata::deepseek_v4 {
#include "pp_plan_receipt_backend.inc"
}
