#include "device_cache.hpp"
#include "cuda_pool_retirement.hpp"
#include "device_cache_budget.hpp"
#include "cache_growth.hpp"
#include <cstdio>
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
bool pp_worker_parking_enabled(){
    const char* value=std::getenv("STRATA_DSV4_PP_WORKER_PARK");
    if(!value)return true;
    if(!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::runtime_error("PP worker parking must be0 or1");
}
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
    model_(model), staging_(staging), cache_(cache), scheduler_(scheduler) {
    if (model.arch != LLM_ARCH_DEEPSEEK4 || (model.layers.size() != 43 && model.layers.size() != 44) || !scheduler)
        throw std::runtime_error("device cache requires the validated DeepSeek4 cache");
    pp_permanent_slots_=int(cache.strata_cost_snapshot().permanent_slots);
    frequency_ = expert_frequency_enabled() ? cache.strata_frequency() : nullptr;
    if (expert_frequency_enabled() && !frequency_) throw std::runtime_error("canonical frequency authority is absent");
    const char* graph_opt = std::getenv("GGML_CUDA_GRAPH_OPT");
    if (graph_opt && std::atoi(graph_opt) == 1)
        throw std::runtime_error("one mapped miss packet requires sequential CUDA lookup scheduling");
    backend_ = cache.backend(0);
    if (!backend_) throw std::runtime_error("device cache has no selected backend");
    try {
        context_ = ggml_init({48 * ggml_tensor_overhead(), nullptr, true});
        if (!context_) throw std::runtime_error("device cache metadata allocation failed");
        directory_ = ggml_new_tensor_1d(context_, GGML_TYPE_I32, device_cache_max_layers * 256);
        status_ = ggml_new_tensor_1d(context_, GGML_TYPE_I32, 1);
        counters_ = ggml_new_tensor_1d(context_, GGML_TYPE_I64, device_cache_counter_count);
        frequency_host_.resize(model.layers.size()*256+1);
        if (frequency_) frequency_gpu_ = ggml_new_tensor_1d(context_, GGML_TYPE_I32, frequency_host_.size());
        for (int layer = 0; layer < int(model_.layers.size()); ++layer) {
            originals_[layer] = cache.strata_directory(layer);
            const auto* original = originals_[layer];
            const auto* bank = cache.get_experts(model.layers[layer].ffn_up_exps);
            if (cache.backend(layer) != backend_ || !original || !bank || !bank->data ||
                original->type != GGML_TYPE_I32 || ggml_nelements(original) != 256 ||
                !ggml_is_contiguous(original) || !original->data || !original->buffer ||
                !ggml_backend_buffer_is_host(original->buffer) || bank->ne[2] < 6)
                throw std::runtime_error("device cache directory/bank contract differs");
            maps_[layer] = ggml_view_2d(context_, directory_, 1, 256, sizeof(int32_t), layer * 256 * sizeof(int32_t));
            ggml_format_name(maps_[layer], "strata_device_cache.map-%d", layer);
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend_);
        if (!buffer_) throw std::runtime_error("device cache directory allocation failed");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const int32_t empty = 0;
        const std::array<uint64_t, device_cache_counter_count> counters{};
        ggml_backend_tensor_set(status_, &empty, 0, sizeof(empty));
        ggml_backend_tensor_set(counters_, counters.data(), 0, sizeof(counters));
        if (frequency_) {
            frequency_->export_state(frequency_host_.data(), frequency_host_.size());
            ggml_backend_tensor_set(frequency_gpu_, frequency_host_.data(), 0, frequency_host_.size()*sizeof(uint32_t));
        }
        dma_ = std::make_unique<DeviceCacheDma>(backend_, staging);
        const int headroom_slots=pp_cache_slots_with_headroom(pp_permanent_slots_);
        if(headroom_slots!=pp_permanent_slots_){cache_.strata_resize_phase(headroom_slots,upload,this);pp_permanent_slots_=headroom_slots;}
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
        project_directory();
        worker_ = std::thread(&DeviceExpertCache::worker, this);
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_.wait(lock, [&] { return worker_ready_; });
        if (!error_.empty()) throw std::runtime_error(error_);
    } catch (...) { release(); throw; }
}
DeviceExpertCache::~DeviceExpertCache() { release(); }
void DeviceExpertCache::release() noexcept {
    if (backend_) ggml_backend_synchronize(backend_);
    stop_.store(true, std::memory_order_release);
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    read_ahead_.reset();cpu_.reset();
    // Drain the copy stream before borrowed source registrations and banks die.
    dma_.reset();
    if (buffer_) { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
    if (context_) { ggml_free(context_); context_ = nullptr; }
}
void DeviceExpertCache::set_enabled(bool enabled) {
    ggml_backend_synchronize(backend_);
    std::lock_guard<std::mutex> lock(service_mutex_);
    if (active_ && frequency_) read_frequency();
    enabled_ = enabled;
    active_ = false;
    worker_active_.store(false, std::memory_order_release);
    dma_->set_active(false);
}
void DeviceExpertCache::prepare(bool single_token) {
    const bool next = enabled_ && single_token;
    if(active_&&!next){
        ggml_backend_synchronize(backend_);
        worker_active_.store(false,std::memory_order_release);
        {std::lock_guard<std::mutex> lock(service_mutex_);
            if(cpu_){cpu_->wait();cpu_->set_spin_active(!pp_worker_parking_enabled());}}
    }
    if (active_ && !next && frequency_) {
        std::lock_guard<std::mutex> lock(service_mutex_);
        if(!pending_admission_.empty()) {
            dma_->publish_native_slots(true);
            cache_.strata_commit_background(pending_admission_);pending_admission_.clear();
        }
        read_frequency();
        if (pp_permanent_slots_ && cache_.strata_cost_snapshot().permanent_slots != uint64_t(pp_permanent_slots_)) {
            cache_.strata_resize_phase(pp_permanent_slots_,upload,this);
            project_directory();
        }
    }
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
        if (frequency_) {
            frequency_->export_state(frequency_host_.data(), frequency_host_.size());
            ggml_backend_tensor_set(frequency_gpu_, frequency_host_.data(), 0, frequency_host_.size()*sizeof(uint32_t));
        }
        project_directory();
    }
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
    result.host_services = services_;
    result.service_wall_ns = service_ns_;
    result.service_max_ns = service_max_ns_;
    result.directory_h2d_bytes = directory_bytes_;
    result.transport = dma_->metrics();
    result.dma_costs = dma_->costs();
    result.planning_ns = planning_ns_;
    result.projection_ns = projection_ns_;
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
    using Read = bool (*)(ggml_backend_t, PpStreamMetrics*);
    const auto read = reinterpret_cast<Read>(ggml_backend_reg_get_proc_address(reg, "strata_pp_stream_metrics"));
    if (read) read(backend_, &result.pp);
    result.capacity = cache_.strata_cost_snapshot();
    trace_device_memory_budget(backend_,active_?"tg_snapshot":"pp_snapshot",int(result.capacity.permanent_slots));
    if(read_ahead_)result.read_ahead=read_ahead_->metrics();
    result.background_bytes=background_bytes_;result.background_commits=background_commits_;
    result.background_staged_bytes=background_staged_bytes_;result.background_staged_experts=background_staged_experts_;
    result.cpu_jobs = cpu_jobs_.load(); result.cpu_routes = cpu_routes_.load();
    result.cpu_logical_weight_bytes = cpu_bytes_.load(); result.cpu_compute_ns = cpu_ns_.load();
    return result;
}
ggml_tensor* DeviceExpertCache::slot_map(int32_t layer, int64_t tokens, int64_t used,
                                       ggml_tensor* original, void* data) {
    auto& self = *static_cast<DeviceExpertCache*>(data);
    self.graph_active_ = self.active_ && (tokens == 1 || (tokens == 2 && native_tg_cpu_only()));
    if (!self.graph_active_) return original;
    if (layer < 0 || layer >= int(self.model_.layers.size()) || used != 6 || original != self.originals_[layer])
        throw std::runtime_error("device cache lookup shape differs");
    return self.maps_[layer];
}
void DeviceExpertCache::decorate(ggml_tensor* tensor, int32_t layer, void* data) {
    auto& self = *static_cast<DeviceExpertCache*>(data);
    if (!self.graph_active_) return;
    if (layer < 0 || layer >= int(self.model_.layers.size()) || tensor->op != GGML_OP_GET_ROWS || tensor->src[0] != self.maps_[layer] ||
        !tensor->src[1] || (ggml_nelements(tensor->src[1]) != 6 && ggml_nelements(tensor->src[1]) != 12) || !ggml_is_contiguous(tensor->src[1]))
        throw std::runtime_error("device cache GET_ROWS seam differs");
    tensor->src[2] = self.status_;
    tensor->src[3] = self.counters_;
    tensor->src[4] = self.frequency_gpu_;
    const uintptr_t packet = self.dma_->device_packet();
    const uint64_t timeout = self.dma_->timeout_ns();
    static_assert(sizeof(packet) <= 2 * sizeof(int32_t));
    std::memcpy(tensor->op_params, &packet, sizeof(packet));
    tensor->op_params[2] = layer;
    std::memcpy(tensor->op_params + 4, &timeout, sizeof(timeout));
    tensor->op_params[device_cache_tag_index] = device_cache_lookup_tag;
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
void DeviceExpertCache::idle() {
    const char* enabled=std::getenv("STRATA_DSV4_TG_BACKGROUND_ADMISSION");
    if(!active_||!native_tg_cpu_only()||!enabled||std::strcmp(enabled,"1"))return;
    const char* stage=std::getenv("STRATA_DSV4_BG_STAGE_UNPINNED");
    const bool staged=stage&&!std::strcmp(stage,"1");
    // Caller has read completed logits, hence no GPU readers or CPU request
    // remains. Never wait for outstanding weight DMA on the next token.
    std::lock_guard<std::mutex> lock(service_mutex_);
    if(!pending_admission_.empty()) {
        if(!dma_->background_ready())return;
        cache_.strata_commit_background(pending_admission_);pending_admission_.clear();
        project_directory();++background_commits_;
    }
    const char* cadence=std::getenv("STRATA_DSV4_BG_ADMISSION_PERIOD");
    unsigned period=16;
    if(cadence){char* end=nullptr;const long value=std::strtol(cadence,&end,10);
        if(!*cadence||*end||value<1||value>64)throw std::runtime_error("background cadence outside1..64");period=unsigned(value);}
    if(++idle_tokens_%period)return;
    if(background_eligible_.empty()||background_staging_mode_!=staged) {
        background_staging_mode_=staged;
        background_eligible_.resize(int(model_.layers.size())*256,1);
        for(int key=0;key<int(model_.layers.size())*256;++key){const auto& layer=model_.layers[key/256];
            background_eligible_[key]=1;
            for(auto* source:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps}) {
                const auto part=staging_?staging_->transport_segment(static_cast<const char*>(source->data)+(key%256)*source->nb[2],source->nb[2]):ExpertSourcePin::Segment{0,false};
                if((!part.registered||part.bytes<source->nb[2])&&!(staged&&key/256<43))background_eligible_[key]=0;
            }
        }
    }
    read_frequency();pending_admission_=cache_.strata_plan_background(background_eligible_.data());
    if(pending_admission_.empty())return;
    for(const auto& item:pending_admission_)if(item.key/256<43){
        const auto& layer=model_.layers[item.key/256];bool needs_stage=false;
        for(auto* source:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps}){
            const auto part=staging_?staging_->transport_segment(static_cast<const char*>(source->data)+(item.key%256)*source->nb[2],source->nb[2]):ExpertSourcePin::Segment{0,false};
            needs_stage|=!part.registered||part.bytes<source->nb[2];
        }
        if(needs_stage)++background_staged_experts_;
    }
    const auto staged_before=dma_->metrics().host_copy_bytes;
    background_bytes_+=cache_.strata_upload_background(pending_admission_,upload,this);
    background_staged_bytes_+=dma_->metrics().host_copy_bytes-staged_before;
    dma_->seal_background();
}
}

namespace strata::deepseek_v4 {
void DeviceExpertCache::write_routing_seed(const std::string& path) const {
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
