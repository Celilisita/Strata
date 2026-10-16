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
    uint64_t prefetch_bytes=0,prefetch_consumed_bytes=0,prefetch_launches=0;
};
PpStreamMetrics pp_stream_backend_metrics(ggml_backend_t);
}
