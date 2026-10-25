#pragma once
#include "strata/dense/spec.hpp"
#include "strata/dense/half_cache.hpp"
#include "strata/dense/projection_pack.hpp"
#include "ggml.h"
#include <unordered_map>
#include <vector>

namespace strata::dense {
struct LayerState {
    ggml_tensor *key = nullptr, *value = nullptr;
    ggml_tensor *history = nullptr, *recurrent = nullptr;
};

// Model weights and persistent state outlive every request graph.
struct GraphContext {
    ggml_context *ctx;
    ggml_cgraph *graph;
    const Spec &spec;
    const std::unordered_map<const TensorInfo *, ggml_tensor *> &weights;
    std::vector<LayerState> &states;
    const HalfCache *pp_half = nullptr;
    const ProjectionPack *projection_pack = nullptr;
    mutable ProjectionPack::Cache projection_cache{};
    ggml_tensor *gdn_norm_scale = nullptr;
    int gdn_norm_fused_nodes = 0;
    ggml_tensor *positions = nullptr, *mask = nullptr, *cache_rows = nullptr;
    int start = 0, tokens = 0, kv_tokens = 0;
    ggml_tensor *w(const TensorInfo *tensor) const { return weights.at(tensor); }
    ggml_tensor *mm(const TensorInfo *tensor, ggml_tensor *input) const {
        if (projection_pack && tokens == 1)
            if (auto *packed = projection_pack->project(ctx,tensor,input,projection_cache)) return packed;
        auto *weight = w(tensor);
        if (pp_half && input->ne[1] > 1)
            if (auto *cached = pp_half->tensor(tensor)) weight = cached;
        return ggml_mul_mat(ctx,weight,input);
    }
    ggml_tensor *norm(ggml_tensor *input, const TensorInfo *weight) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, input, float(spec.rms_epsilon)), w(weight));
    }
};

ggml_tensor *build_recurrent(GraphContext &, const LayerTensors &, LayerState &, ggml_tensor *normalized);
ggml_tensor *build_attention(GraphContext &, const LayerTensors &, LayerState &, ggml_tensor *normalized);
ggml_tensor *build_ffn(GraphContext &, const LayerTensors &, ggml_tensor *normalized);
} // namespace strata::dense
