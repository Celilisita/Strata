#pragma once
#include <cstdlib>
#include <cstring>

// Keep the original activation and route IDs alive until the existing fused
// Gate/Up/GLU finishes. This changes allocator liveness, not model computation.
// Called only while the canonical backend plans allocation dependencies.
inline void strata_glu_alloc_dependencies(ggml_cgraph* graph,
                                         ggml_backend_graph_optimize_params* params) {
    static const bool enabled = [] {
        const char* option = std::getenv("STRATA_DSV4_GLU_ALLOC_DEPS");
        return option && std::strcmp(option, "1") == 0;
    }();
    if (!enabled || ggml_cuda_info().devices[ggml_cuda_get_device()].cc != 1200) return;
    for (int i = 0; i + 2 < graph->n_nodes; ++i) {
        auto* first = graph->nodes[i];
        const bool routed = first->op == GGML_OP_MUL_MAT_ID;
        if (!routed && first->op != GGML_OP_MUL_MAT) continue;
        const ggml_op ops[] = {first->op, first->op, GGML_OP_GLU};
        const int outputs[] = {i + 2};
        if (!ggml_can_fuse_subgraph(graph, i, 3, ops, outputs, 1)) continue;
        auto* glu = graph->nodes[i + 2];
        auto* gate = glu->src[0];
        auto* up = glu->src[1];
        const bool pair = (gate == first && up == graph->nodes[i + 1]) ||
                          (up == first && gate == graph->nodes[i + 1]);
        if (!pair || !ggml_cuda_should_fuse_mul_mat(up, gate, glu) ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU_CLAMP ||
            ggml_get_op_params_f32(glu, 3) != 10.0f) continue;
        const auto* weight = up->src[0];
        auto* input = up->src[1];
        const int64_t tokens = routed ? glu->ne[2] : glu->ne[1];
        if (weight->type != (routed ? GGML_TYPE_MXFP4 : GGML_TYPE_Q8_0) ||
            weight->ne[0] != 4096 || weight->ne[1] != 2048 ||
            weight->ne[3] != 1 || (!routed && weight->ne[2] != 1) ||
            (tokens != 1 && !(routed && tokens == 5)) ||
            input->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(input) || !ggml_is_contiguous(glu)) continue;
        params->add_alloc_dep(params->user_data, input, glu);
        if (routed) params->add_alloc_dep(params->user_data, up->src[2], glu);
    }
}
