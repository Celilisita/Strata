#include "strata/dense/graph.hpp"
#include <cmath>
#include <stdexcept>

namespace strata::dense {
ggml_tensor *build_attention(GraphContext &g, const LayerTensors &l, LayerState &state,
                             ggml_tensor *input) {
    auto *ctx = g.ctx;
    const auto &s = g.spec;
    const int64_t d = s.head_dim, heads = s.n_head, kvheads = s.n_head_kv;
    auto *qfull = g.mm(l.q, input);
    auto *k = ggml_reshape_3d(ctx, g.mm(l.k, input), d, kvheads, g.tokens);
    auto *v = ggml_reshape_3d(ctx, g.mm(l.v, input), d, kvheads, g.tokens);
    ggml_tensor *q = nullptr, *gate = nullptr;
    if (s.architecture == "qwen35") {
        // Query and gate are interleaved per head, not two global halves.
        q = ggml_view_3d(ctx, qfull, d, heads, g.tokens, 2*d*sizeof(float),
                         2*d*heads*sizeof(float), 0);
        gate = ggml_view_3d(ctx, qfull, d, heads, g.tokens, 2*d*sizeof(float),
                            2*d*heads*sizeof(float), d*sizeof(float));
        gate = ggml_cont_2d(ctx, gate, d*heads, g.tokens);
        q = g.norm(q, l.q_norm);
        k = g.norm(k, l.k_norm);
        int sections[4];
        for (int i = 0; i < 4; ++i) sections[i] = int(s.rope_sections[i]);
        q = ggml_rope_multi(ctx, q, g.positions, nullptr, int(s.rope_dim), sections,
                            GGML_ROPE_TYPE_IMROPE, int(s.context_length), float(s.rope_base),
                            1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_multi(ctx, k, g.positions, nullptr, int(s.rope_dim), sections,
                            GGML_ROPE_TYPE_IMROPE, int(s.context_length), float(s.rope_base),
                            1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    } else {
        q = ggml_reshape_3d(ctx, qfull, d, heads, g.tokens);
        q = ggml_rope_ext(ctx, q, g.positions, nullptr, int(s.rope_dim), GGML_ROPE_TYPE_NORMAL,
                          int(s.context_length), float(s.rope_base), 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(ctx, k, g.positions, nullptr, int(s.rope_dim), GGML_ROPE_TYPE_NORMAL,
                          int(s.context_length), float(s.rope_base), 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }
    const size_t row = size_t(d*kvheads)*sizeof(ggml_fp16_t);
    auto *key_source = ggml_reshape_2d(ctx, k, d*kvheads, g.tokens);
    auto *value_source = ggml_reshape_2d(ctx, v, d*kvheads, g.tokens);
    // Both writes precede FA's cache reads on the backend's ordered stream.
    // Indexed TG writes keep all graph addresses/parameters fixed in a KV bucket.
    if (g.cache_rows) {
        if (g.tokens != 1 || g.cache_rows->type != GGML_TYPE_I64)
            throw std::invalid_argument("dense indexed cache writes require one I64 row");
        ggml_build_forward_expand(g.graph, ggml_set_rows(ctx, state.key, key_source, g.cache_rows));
        ggml_build_forward_expand(g.graph, ggml_set_rows(ctx, state.value, value_source, g.cache_rows));
    } else {
        auto *key_rows = ggml_view_2d(ctx, state.key, d*kvheads, g.tokens, row, size_t(g.start)*row);
        auto *value_rows = ggml_view_2d(ctx, state.value, d*kvheads, g.tokens, row, size_t(g.start)*row);
        ggml_build_forward_expand(g.graph, ggml_cpy(ctx, key_source, key_rows));
        ggml_build_forward_expand(g.graph, ggml_cpy(ctx, value_source, value_rows));
    }
    auto *keys = ggml_view_3d(ctx, state.key, d, kvheads, g.kv_tokens,
                             size_t(d)*sizeof(ggml_fp16_t), row, 0);
    auto *values = ggml_view_3d(ctx, state.value, d, kvheads, g.kv_tokens,
                               size_t(d)*sizeof(ggml_fp16_t), row, 0);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    keys = ggml_permute(ctx, keys, 0, 2, 1, 3);
    values = ggml_permute(ctx, values, 0, 2, 1, 3);
    auto *out = ggml_flash_attn_ext(ctx, q, keys, values, g.mask, 1.0f/std::sqrt(float(d)), 0, 0);
    out = ggml_reshape_2d(ctx, out, d*heads, g.tokens);
    if (gate) out = ggml_mul(ctx, out, ggml_sigmoid(ctx, gate));
    return g.mm(l.attn_out, out);
}

ggml_tensor *build_ffn(GraphContext &g, const LayerTensors &l, ggml_tensor *input) {
    auto *gate = g.mm(l.ffn_gate, input);
    auto *up = g.mm(l.ffn_up, input);
    // Keep the two projections adjacent to a single SwiGLU operation. This
    // exposes the backend's joint gate/up GEMV path without changing the math.
    auto *activation = ggml_swiglu_split(g.ctx, gate, up);
    return g.mm(l.ffn_down, activation);
}
} // namespace strata::dense
