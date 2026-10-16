#pragma once
#include "device_cache_trace.hpp"
#include "ggml-cuda/common.cuh"

namespace strata::deepseek_v4 {
inline void cache_graph_preupload(cudaGraphExec_t graph, cudaStream_t stream, bool changed) {
    auto* trace = cache_graph_current_trace();
    if (!changed || !trace || !trace->graph_misses.load(std::memory_order_acquire)) return;
    // First launch otherwise uploads resources while an early lookup may
    // already await DMA. WDDM can hold that DMA call until the lookup expires.
    // Upload executes no graph nodes; complete preparation before any lookup.
    const auto begin = cache_trace_nanoseconds();
    CUDA_CHECK(cudaGraphUpload(graph, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    trace->graph_preuploads.fetch_add(1, std::memory_order_relaxed);
    trace->graph_preupload_ns.fetch_add(cache_trace_nanoseconds() - begin, std::memory_order_relaxed);
}
}
