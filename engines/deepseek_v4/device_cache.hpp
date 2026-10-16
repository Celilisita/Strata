#pragma once
#include "device_cache_backend.hpp"
#include "pp_stream_metrics.hpp"
#include "cache_cost_audit.hpp"
#include "mxfp4_cpu_service.hpp"
#include "native_tg_policy.hpp"
#include "cache_background.hpp"
#include "native_read_ahead.hpp"
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
struct DeviceCacheMetrics {
    std::array<uint64_t, device_cache_counter_count> gpu{};
    uint64_t host_services = 0, directory_h2d_bytes = 0, service_wall_ns = 0, service_max_ns = 0;
    uint64_t planning_ns = 0, projection_ns = 0;
    ExpertStagingMetrics transport;
    DeviceCacheDmaCosts dma_costs;
    PpStreamMetrics pp;
    CacheCostSnapshot capacity;
    uint64_t cpu_jobs = 0, cpu_routes = 0, cpu_logical_weight_bytes = 0, cpu_compute_ns = 0;
    uint64_t background_bytes=0,background_commits=0,background_staged_bytes=0,background_staged_experts=0;
    NativeReadAheadMetrics read_ahead;
};
// Canonical cache owns all banks and residency. GPU directories are projections;
// only a real miss invokes its host planner. PP and TG share the same cache.
class DeviceExpertCache {
public:
    DeviceExpertCache(const llama_model&, llama_moe_cache&, ggml_backend_sched_t, ExpertStaging*);
    ~DeviceExpertCache();
    void prepare(bool single_token);
    bool phase_change(bool single_token) const { return enabled_ && single_token != active_; }
    void set_enabled(bool enabled);
    void verify() const;
    void idle();
    Mxfp4CpuService& cpu_service() { if(!cpu_)throw std::runtime_error("native CPU pool absent");return *cpu_; }
    void write_routing_seed(const std::string&) const;
    DeviceCacheMetrics metrics() const;
    static ggml_tensor* slot_map(int32_t, int64_t, int64_t, ggml_tensor*, void*);
    static void decorate(ggml_tensor*, int32_t, void*);
    static ggml_tensor* guard(ggml_context*, ggml_tensor*, void*);
private:
    void release() noexcept;
    void project_directory();
    void read_frequency();
    void worker();
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
    ExpertFrequency* frequency_ = nullptr;
    std::vector<uint32_t> frequency_host_;
    std::array<ggml_tensor*, device_cache_max_layers> maps_{}, originals_{};
    std::array<int32_t, device_cache_max_layers * 256> directory_host_{};
    std::unique_ptr<DeviceCacheDma> dma_;
    std::unique_ptr<Mxfp4CpuService> cpu_;
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
    std::vector<uint8_t> background_eligible_;
    bool background_staging_mode_=false;
    BackgroundAdmissions pending_admission_;
    uint64_t idle_tokens_=0,background_bytes_=0,background_commits_=0,background_staged_bytes_=0,background_staged_experts_=0;
    int pp_permanent_slots_ = 0;
    bool worker_ready_ = false;
    std::string error_;
    bool enabled_ = true, active_ = false, graph_active_ = false;
};
void configure_device_expert_cache(llama_context*);
void prepare_device_expert_cache(llama_context*, size_t tokens);
void idle_device_expert_cache(llama_context*);
ggml_backend_t device_expert_backend(llama_context*);
void write_device_routing_seed(llama_context*,const std::string&);
void enable_device_expert_cache(llama_context*, bool enabled);
void verify_device_expert_cache(llama_context*);
DeviceCacheMetrics device_expert_cache_metrics(llama_context*);
bool device_expert_cache_requested();
void configure_device_cache_process();
}
