#pragma once

#include "strata/dense/model.hpp"

namespace strata::dense {
// Owns graph metadata and reusable host inputs. Device workspace is borrowed
// from Model's one gallocr, so every alternate allocation invalidates this graph.
struct RequestGraph {
    ggml_context *context = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *tokens = nullptr, *positions = nullptr, *mask = nullptr;
    ggml_tensor *cache_rows = nullptr, *logits = nullptr;
    ggml_tensor *greedy_id = nullptr, *finite_sum = nullptr;
    ggml_tensor *image_embeddings = nullptr, *image_rows = nullptr;
    int count = 0, kv = 0, axes = 0;
    bool indexed_write = false, device_greedy = false;
    int projection_fused_matmuls = 0;
    int norm_fused_nodes = 0;
    std::vector<int> host_positions;
    std::vector<ggml_fp16_t> host_mask;
    int64_t host_cache_row = 0;

    ~RequestGraph() { if (context) ggml_free(context); }
    RequestGraph() = default;
    RequestGraph(const RequestGraph &) = delete;
    RequestGraph &operator=(const RequestGraph &) = delete;
};
} // namespace strata::dense
