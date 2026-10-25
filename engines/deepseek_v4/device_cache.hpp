#pragma once
#include "device_cache_backend.hpp"
#include "pp_stream_metrics.hpp"
#include "pp_plan_receipt.hpp"
#include "cache_cost_audit.hpp"
#include "mxfp4_cpu_service.hpp"
#include "native_tg_policy.hpp"
#include "cache_background.hpp"
#include "native_read_ahead.hpp"
#include "../glm5_next/hybrid_four_scratch.hpp"
#include "../glm5_next/phase_growth.hpp"
#include "../glm5_next/cpu_gu_q8_metrics.hpp"
#include "../glm5_next/causal_phase_host.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <stdexcept>

struct llama_model;
struct llama_context;
class llama_moe_cache;
namespace strata::deepseek_v4 {
class ExpertFrequency;
}
namespace strata::glm5_next {class HybridProvider;class RouteCapture;class ColdGpuOwner;}
namespace strata::deepseek_v4 {
struct DeviceCacheMetrics {
    std::array<uint64_t, device_cache_counter_count> gpu{};
    uint64_t host_services = 0, directory_h2d_bytes = 0, service_wall_ns = 0, service_max_ns = 0;
    uint64_t planning_ns = 0, projection_ns = 0;
    uint64_t phase_transitions=0,phase_resize_ns=0,phase_seed_ns=0,phase_seed_bytes=0,phase_verify_bytes=0,phase_evicted_keys=0;
    ExpertStagingMetrics transport;
    DeviceCacheDmaCosts dma_costs;
    PpStreamMetrics pp;
    const PpPlanReceiptPod* pp_plan_receipt=nullptr; // Borrowed only after explicit request retirement.
    CacheCostSnapshot capacity;
    strata::glm5_next::PhaseCapacity glm_capacity;
    std::array<uint64_t,4> prefill_coverage{};
    std::array<uint64_t,7> last_phase_budget{};
    uint64_t activation_control_d2h_bytes=0,cpu_contribution_h2d_bytes=0,hybrid_gpu_services=0;
    uint64_t hybrid_overlapped_services=0;
    // 0..5 cumulative: validated jobs, occurrences, logical copy bytes,
    // hooked mask0 jobs, reserve attempts, capacity fallbacks; 6..7 gauges.
    std::array<uint64_t,8> cold_gpu{};
    uint64_t cold_gpu_reserved_rows=0,cold_gpu_metadata_bytes=0;
    uint64_t cold_gpu_two_jobs=0,cold_gpu_four_jobs=0,cold_gpu_requested_max_rows=2;
    std::array<uint64_t,4> cold_gpu_topology{};
    std::array<uint64_t,3> cold_gpu_repeat_priority{};
    uint32_t cold_gpu_topology_requested=0;
    std::array<uint64_t,5> cold_gpu_conditional4{};
    bool cold_gpu_conditional4_requested=false,cold_gpu_conditional4_active=false;
    strata::glm5_next::CausalHostSnapshot causal_host;
    uint64_t cpu_jobs = 0, cpu_routes = 0, cpu_logical_weight_bytes = 0, cpu_compute_ns = 0;
    uint64_t background_bytes=0,background_commits=0,background_staged_bytes=0,background_staged_experts=0;
    uint64_t background_verify_bytes=0;
    int cpu_service_threads=0;
    strata::glm5_next::CpuGuQ8PreparationMetrics cpu_gu_q8_preparation;
    std::array<uint64_t,17*17> cpu_pair_jobs{},cpu_pair_ns{};
    std::array<uint64_t,33*33> cpu_four_jobs{},cpu_four_ns{};
    std::array<uint64_t,3> cpu_four_capacity_before{},cpu_four_capacity_after{};
    double cpu_four_prepare_ms=0;
    uint64_t cpu_four_scratch_bytes=0;
    std::array<uint64_t,4> background_layout_commits{};
    NativeReadAheadMetrics read_ahead;
    std::array<std::array<int32_t, device_cache_max_routes>, device_cache_max_layers> last_miss_ids{};
    std::array<uint32_t, device_cache_max_layers> last_miss_counts{};
};
// Canonical cache owns all banks and residency. GPU directories are projections;
// only a real miss invokes its host planner. PP and TG share the same cache.
class DeviceExpertCache {
public:
    DeviceExpertCache(const llama_model&, llama_moe_cache&, ggml_backend_sched_t, ExpertStaging*);
    ~DeviceExpertCache();
    void prepare(bool single_token);
    // Explicit one-cohort diagnostic; requires a prepared TG owner and no live graph.
    void prepare_partial_down_diagnostic(uint64_t generation=1);
    bool refresh_cold_gpu_rows();
    ggml_backend_sched_t borrow_scheduler(ggml_backend_sched_t value) {
        if(!value)throw std::runtime_error("cache graph scheduler cannot be null");
        auto* previous=scheduler_;scheduler_=value;return previous;
    }
    bool phase_change(bool single_token);
    void set_graph_retirement(void(*fn)(void*),void* user){borrowed_retire_=fn;borrowed_retire_user_=user;}
    void set_enabled(bool enabled);
    void verify() const;
    void idle();
    void settle_background();
    Mxfp4CpuService& cpu_service() { if(!cpu_)throw std::runtime_error("native CPU pool absent");return *cpu_; }
    void write_routing_seed(const std::string&) const;
    void begin_pp_plan_receipt(uint64_t,bool);
    const PpPlanReceiptPod* read_pp_plan_receipt() const;
    void begin_route_capture();
    void write_route_capture(const std::string&);
    void phase_probe_reverse_prior();
    std::array<uint64_t,4> probe_background();
    DeviceCacheMetrics metrics() const;
    static ggml_tensor* slot_map(int32_t, int64_t, int64_t, ggml_tensor*, void*);
    static void decorate(ggml_tensor*, int32_t, void*);
    static ggml_tensor* guard(ggml_context*, ggml_tensor*, void*);
private:
    void release() noexcept;
    void project_directory();
    void read_frequency();
    bool finish_background(bool wait);
    void worker();
    void service_hybrid(int32_t,strata::glm5_next::CausalHostSample* causal=nullptr);
    void service_hybrid_four(int32_t);
    void glm_phase_transition(bool to_tg);
    void glm_capacity_admit(size_t);
    bool glm_capacity_due(bool);
    void glm_capacity_recheck();
    uint32_t cold_gpu_mask(int32_t,int,uint32_t,const int32_t*);
    static bool upload(ggml_backend_t, ggml_tensor*, const void*, size_t, size_t, void*);
    void service_pair(int32_t layer, uint32_t ticket);
    void service_native(int32_t layer, const std::array<int32_t, 6>& ids, uint32_t ticket);
    static void cpu_completed(void*, bool);
    const llama_model& model_;
    ExpertStaging* staging_ = nullptr;
    llama_moe_cache& cache_;
    ggml_backend_sched_t scheduler_;
    ggml_backend_t backend_ = nullptr;
    ggml_context* context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor *directory_ = nullptr, *status_ = nullptr, *counters_ = nullptr;
    ggml_tensor* frequency_gpu_ = nullptr;
    ggml_tensor* hybrid_ticket_ = nullptr;
    ExpertFrequency* frequency_ = nullptr;
    std::vector<uint32_t> frequency_host_;
    std::array<ggml_tensor*, device_cache_max_layers> maps_{}, originals_{};
    std::vector<int32_t> directory_host_;
    int experts_ = 256, top_k_ = 6;
    uint64_t owner_epoch_ = 0;
    uint64_t pp_receipt_nonce_=0;bool pp_receipt_available_=false;
    std::array<std::array<int32_t, device_cache_max_routes>, device_cache_max_layers> last_miss_ids_{};
    std::array<uint32_t, device_cache_max_layers> last_miss_counts_{};
    std::unique_ptr<DeviceCacheDma> dma_;
    std::unique_ptr<Mxfp4CpuService> cpu_;
    std::unique_ptr<strata::glm5_next::HybridProvider> hybrid_;
    std::unique_ptr<strata::glm5_next::ColdGpuOwner> cold_gpu_;
    std::unique_ptr<strata::glm5_next::HybridFourScratch> hybrid_four_scratch_;
    std::unique_ptr<strata::glm5_next::RouteCapture> route_capture_;
    uint64_t activation_control_bytes_=0,contribution_bytes_=0,hybrid_gpu_services_=0;
    uint64_t hybrid_overlapped_services_=0;
    strata::glm5_next::CausalHostStats causal_host_;
    bool hybrid_active_=false;
    std::unique_ptr<NativeReadAhead> read_ahead_;
    uint32_t (*native_ready_)(const DeviceMissPacket*) = nullptr;
    void (*native_reply_)(DeviceMissPacket*, uint32_t) = nullptr;
    void (*native_complete_)(DeviceMissPacket*, uint32_t, bool) = nullptr;
    NativeTgPolicy native_policy_;
    uintptr_t native_slots_ = 0;
    uint32_t cpu_ticket_ = 0;
    size_t cpu_count_ = 0;
    uint64_t cpu_started_ns_ = 0;
    std::atomic<uint64_t> cpu_jobs_{0}, cpu_routes_{0}, cpu_bytes_{0}, cpu_ns_{0};
    std::array<std::atomic<uint64_t>, 7> cpu_sample_ns_{};
    std::atomic<bool> stop_{false}, worker_active_{false};
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    mutable std::mutex service_mutex_;
    uint64_t services_ = 0, directory_bytes_ = 0, service_ns_ = 0, service_max_ns_ = 0;
    uint64_t planning_ns_ = 0, projection_ns_ = 0;
    uint64_t phase_transitions_=0,phase_resize_ns_=0,phase_seed_ns_=0,phase_seed_bytes_=0,phase_verify_bytes_=0,phase_evicted_keys_=0;
    bool glm_tg_phase_=false,glm_capacity_pending_=false,glm_capacity_poisoned_=false;
    uint64_t glm_tg_calls_=0;
    void(*borrowed_retire_)(void*)=nullptr;void* borrowed_retire_user_=nullptr;
    strata::glm5_next::PhaseCapacity glm_capacity_;
    std::array<uint64_t,7> last_phase_budget_{};
    std::vector<uint8_t> background_eligible_;
    bool background_staging_mode_=false;
    uint64_t background_verify_bytes_=0;
    std::array<uint64_t,4> background_layout_commits_{};
    BackgroundAdmissions pending_admission_;
    uint64_t idle_tokens_=0,background_bytes_=0,background_commits_=0,background_staged_bytes_=0,background_staged_experts_=0;
    int pp_permanent_slots_ = 0;
    bool worker_ready_ = false;
    std::string error_;
    bool enabled_ = true, active_ = false, graph_active_ = false;
};
void register_device_cache_graph_retirement(llama_context*,void(*)(void*),void*);
void configure_device_expert_cache(llama_context*);
void prepare_device_expert_cache(llama_context*, size_t tokens);
void idle_device_expert_cache(llama_context*);
void settle_device_background(llama_context*);
ggml_backend_t device_expert_backend(llama_context*);
void write_device_routing_seed(llama_context*,const std::string&);
void begin_device_pp_plan_receipt(llama_context*,uint64_t,bool);
const PpPlanReceiptPod* read_device_pp_plan_receipt(llama_context*);
void begin_device_route_capture(llama_context*);
void write_device_route_capture(llama_context*,const std::string&);
void reverse_phase_probe_prior(llama_context*);
std::array<uint64_t,4> probe_device_background(llama_context*);
void enable_device_expert_cache(llama_context*, bool enabled);
void verify_device_expert_cache(llama_context*);
DeviceCacheMetrics device_expert_cache_metrics(llama_context*);
bool device_expert_cache_requested();
void configure_device_cache_process();
}
