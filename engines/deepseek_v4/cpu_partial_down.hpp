#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::deepseek_v4 {

// Private, explicit diagnostic seam. No environment flag or normal-path selection.
// All arrays and callback state are borrowed until synchronous compute returns.
// publish_mid borrows contiguous F32[2048,1,10] only during that callback. It must
// copy the selected rows to separately owned packet storage before returning true.
// It may throw: the scheduler adapter catches it before returning through C.
using PartialDownPublish = bool (*)(void*, const float*, std::size_t, std::size_t);

struct PartialDownJob {
    int layer = 24;
    const float* activations = nullptr;
    std::size_t positions_count = 2;
    const int32_t* ids = nullptr;
    const uint8_t* positions = nullptr;
    const uint8_t* original_ranks = nullptr;
    std::size_t gu_routes = 10;
    const uint8_t* down_rows = nullptr;
    std::size_t down_routes = 6;
    float* output = nullptr;
    std::size_t output_elements = 0;
    PartialDownPublish publish_mid = nullptr;
    void* publication_context = nullptr;
    // Caller publishes partialdecision only after preflight succeeds. After that
    // commit, every failure is a same-ticket error/drain, never stock replay.
    bool partial_decision_committed = false;
};

struct PartialDownReceipt {
    std::size_t gu_routes = 0, down_routes = 0, mid_width = 0;
    uint32_t publication_callbacks = 0;
    bool mid_published = false;
};

struct PartialDownCapacity {
    std::size_t prepared_graphs = 0, graph_nodes = 0, graph_tensors = 0;
    std::size_t graph_context_used_bytes = 0, scheduler_cpu_buffer_bytes = 0;
    std::size_t activation_capacity_bytes = 0;
    std::size_t original_mid_bytes = 0, compact_mid_bytes = 0, output_bytes = 0;
    std::size_t gather_index_bytes = 0, down_id_bytes = 0;
    // Allocation inventory, not a reservation or process peak. CPU backend work
    // buffer high-water, scheduler metadata and other service graphs are excluded.
};

} // namespace strata::deepseek_v4
