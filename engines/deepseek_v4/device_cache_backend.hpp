#pragma once
#include "device_cache_protocol.hpp"
#include "expert_staging.hpp"
#include "device_cache_trace.hpp"
#include <array>
#include <memory>
#include <vector>

namespace strata::glm5_next {struct PartialDownPhase;}
namespace strata::deepseek_v4 {
struct DeviceCacheDmaCosts {
    uint64_t graph_compute_calls=0,graph_captures=0,graph_launches=0;
    bool causal_launch_host_enabled=false;
    uint64_t graph_launch_host_ns=0,graph_launch_host_max_ns=0,graph_launch_host_count=0,graph_launch_host_errors=0;
    uint64_t submit_ns = 0, commit_wait_ns = 0;
    // memcpy, slab event record, directory copy, directory event record,
    // commit event wait, slab event wait, frequency copy/record/wait,
    // bounded eager-queue event record/query.
    // Worst interval of each exact API.
    std::array<CacheApiSpan, 13> api{};
    std::array<uint64_t, cache_resource_count> resource_max_ns{}, resource_begin_ns{}, resource_end_ns{};
    std::array<uint32_t, cache_resource_count> resource_stream{};
    uint64_t enqueue_miss_waits = 0, enqueue_miss_wait_ns = 0, enqueue_miss_deadlines = 0;
    uint64_t frequency_d2h_bytes = 0, frequency_read_ns = 0;
    uint64_t enqueue_budget_records = 0, enqueue_budget_queries = 0;
    uint64_t enqueue_budget_waits = 0, enqueue_budget_wait_ns = 0;
    uint64_t graph_preuploads = 0, graph_preupload_ns = 0;
    uint64_t native_slot_h2d_bytes = 0, native_slot_publish_calls = 0;
};
class DeviceCacheDma {
public:
    explicit DeviceCacheDma(ggml_backend_t backend, ExpertStaging* source_registration,
                           bool prepare_all_streams = true, bool blocking_host_wait = false,
                           bool pack_fragments = false,bool causal_phases = false);
    ~DeviceCacheDma();
    DeviceCacheDma(const DeviceCacheDma&) = delete;
    DeviceCacheDma& operator=(const DeviceCacheDma&) = delete;
    DeviceMissPacket* packet() const noexcept;
    uintptr_t device_packet() const noexcept;
    // Explicit diagnostic only; caller owns full GPU/worker quiescence.
    void prepare_partial_down_diagnostic();
    strata::glm5_next::PartialDownPhase* partial_down_phase() const noexcept;
    uintptr_t partial_down_device_phase() const noexcept;
    uint64_t timeout_ns() const noexcept;
    // Bind once before this thread uploads/commits. Rebinding during a waiting
    // compute kernel can serialize with another thread's context operations.
    void bind_thread();
    void set_active(bool);
    void upload(ggml_tensor* destination, const void* source, size_t offset, size_t bytes);
    void commit(void* device_directory, const int32_t* directory, size_t count);
    template<class Directory> void commit(void* destination, const Directory& directory) {
        commit(destination, directory.data(), directory.size());
    }
    void seal_background();
    bool background_ready();
    void wait_background();
    void publish_native_slots(bool payload_pending = true);
    void read_frequency(const void*, std::vector<uint32_t>&);
    void read_miss_frequency(std::vector<uint32_t>&);
    ExpertStagingMetrics metrics() const noexcept;
    DeviceCacheDmaCosts costs() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
