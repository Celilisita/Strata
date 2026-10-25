#pragma once
#include "ggml-backend.h"
#include <cstdint>
namespace strata::deepseek_v4 {
struct PpStreamMetrics {
    uint64_t layers = 0, groups = 0, payload_bytes = 0, dma_calls = 0, staged_bytes = 0;
    uint64_t weight_window_bytes = 0, activation_scratch_bytes = 0;
    uint64_t grouped_products = 0, route_host_reads = 0;
    uint64_t source_cpu_copy_ns = 0;
    uint64_t cache_reused_bytes = 0, cache_copy_calls = 0;
    uint64_t device_alloc_ns = 0, device_alloc_calls = 0, staging_wait_ns = 0;
    uint64_t timed_layers = 0, gpu_layer_ns = 0, gpu_weight_wait_ns = 0, gpu_products_ns = 0;
    uint64_t gpu_prelude_ns = 0, gpu_intergroup_ns = 0, gpu_tail_ns = 0;
    uint64_t prefetch_bytes=0,prefetch_consumed_bytes=0,prefetch_launches=0,prefetch_verified_bytes=0;
    uint64_t prefetch_enqueue_ns=0,prefetch_gpu_copy_ns=0;
    uint64_t source_planar_groups=0,source_planar_bytes=0,source_planar_extra_GU_products=0;
    uint64_t miss_coalesced_layers=0,miss_coalesced_groups_before=0,miss_coalesced_groups_after=0;
    uint64_t prefix_segment_ready_banks=0,prefix_segment_ready_events=0,prefix_segment_ready_waits=0;
    uint64_t transport_verified_groups=0,transport_verified_bytes=0,transport_verify_D2H_bytes=0;
    bool miss_coalesce_active=false; // Latest run request; counts are successful layer submissions.
    bool source_planar_active=false; // Latest native PP run mode; a gauge, not a delta.
};
PpStreamMetrics pp_stream_backend_metrics(ggml_backend_t);
}
