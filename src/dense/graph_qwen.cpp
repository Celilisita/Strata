#include "strata/dense/graph.hpp"

#include <cmath>
#include <stdexcept>

namespace strata::dense {
namespace {

ggml_tensor *l2_norm(ggml_context *ctx, ggml_tensor *input, float epsilon,
                     ggml_tensor *scale) {
    const float dimension = float(input->ne[0]);
    // GDN uses x / sqrt(sum(x*x) + epsilon), rather than RMS normalization.
    auto *normalized = ggml_rms_norm(ctx, input, epsilon / dimension);
    if (!scale) return ggml_scale(ctx, normalized, 1.0f / std::sqrt(dimension));
    if (scale->type != GGML_TYPE_F32 || scale->ne[0] != input->ne[0] ||
        scale->ne[1] != 1 || scale->ne[2] != 1 || scale->ne[3] != 1 ||
        !ggml_is_contiguous(scale))
        throw std::invalid_argument("invalid persistent GDN norm multiplier");
    // Keep RMS as src0: the pinned CUDA backend permits the constant's row
    // broadcast in this operand order and fuses adjacent RMS_NORM + MUL.
    return ggml_mul(ctx, normalized, scale);
}

} // namespace

ggml_tensor *build_recurrent(GraphContext &g, const LayerTensors &l,
                             LayerState &persistent, ggml_tensor *normalized) {
    auto *ctx = g.ctx;
    const auto &s = g.spec;
    const int64_t key_dim = int64_t(s.ssm_state);
    const int64_t key_heads = int64_t(s.ssm_group);
    const int64_t value_heads = int64_t(s.ssm_rank);
    const int64_t inner = int64_t(s.ssm_inner);
    const int64_t kernel = int64_t(s.ssm_conv);
    const int64_t tokens = g.tokens;
    if (key_dim <= 0 || key_heads <= 0 || value_heads <= 0 ||
        value_heads % key_heads || inner != key_dim * value_heads ||
        kernel < 2 || tokens < 1 || !persistent.history || !persistent.recurrent)
        throw std::invalid_argument("invalid dense GDN geometry or persistent state");

    const int64_t key_width = key_dim * key_heads;
    const int64_t channels = 2 * key_width + inner;
    if (persistent.history->type != GGML_TYPE_F32 ||
        persistent.recurrent->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(persistent.history) ||
        !ggml_is_contiguous(persistent.recurrent) ||
        ggml_nelements(persistent.history) != (kernel - 1) * channels ||
        ggml_nelements(persistent.recurrent) != key_dim * key_dim * value_heads)
        throw std::invalid_argument("invalid dense GDN persistent tensor layout");

    auto *mixed = g.mm(l.qkv, normalized);
    mixed = ggml_reshape_3d(ctx, mixed, channels, tokens, 1);
    auto *z = g.mm(l.gate, normalized);
    z = ggml_reshape_4d(ctx, z, key_dim, value_heads, tokens, 1);

    auto *beta = g.mm(l.beta, normalized);
    beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta, 1, value_heads, tokens, 1));
    auto *alpha = g.mm(l.alpha, normalized);
    alpha = ggml_reshape_3d(ctx, alpha, value_heads, tokens, 1);
    auto *decay = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, g.w(l.dt))),
                           g.w(l.a));
    // GGUF ssm_a already contains -exp(A_log). The fused GDN exponentiates
    // this log decay once for each token.
    decay = ggml_reshape_4d(ctx, decay, 1, value_heads, tokens, 1);

    auto *history = ggml_reshape_3d(ctx, persistent.history, kernel - 1, channels, 1);
    auto *conv_input = ggml_concat(ctx, history, ggml_transpose(ctx, mixed), 0);
    // Convolution consumes a materialized concat. Saving its final history
    // before running the convolution is safe, including when tokens < kernel.
    auto *next_history = ggml_view_3d(ctx, conv_input, kernel - 1, channels, 1,
                                     conv_input->nb[1], conv_input->nb[2],
                                     size_t(tokens) * sizeof(float));
    ggml_build_forward_expand(g.graph, ggml_cpy(ctx, next_history, history));

    auto *convolved = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, g.w(l.conv)));
    const size_t scalar_bytes = ggml_element_size(convolved);
    const size_t head_stride = size_t(key_dim) * scalar_bytes;
    const size_t token_stride = size_t(channels) * scalar_bytes;
    const size_t sequence_stride = token_stride * size_t(tokens);
    auto *q = ggml_view_4d(ctx, convolved, key_dim, key_heads, tokens, 1,
                           head_stride, token_stride, sequence_stride, 0);
    auto *k = ggml_view_4d(ctx, convolved, key_dim, key_heads, tokens, 1,
                           head_stride, token_stride, sequence_stride,
                           size_t(key_width) * scalar_bytes);
    auto *v = ggml_view_4d(ctx, convolved, key_dim, value_heads, tokens, 1,
                           head_stride, token_stride, sequence_stride,
                           size_t(2 * key_width) * scalar_bytes);
    q = l2_norm(ctx, q, float(s.rms_epsilon), g.gdn_norm_scale);
    k = l2_norm(ctx, k, float(s.rms_epsilon), g.gdn_norm_scale);
    if (g.gdn_norm_scale) g.gdn_norm_fused_nodes += 2;

    auto *state = ggml_reshape_4d(ctx, persistent.recurrent,
                                 key_dim, key_dim, value_heads, 1);
    // The operator includes the 1/sqrt(key_dim) query scale. K=1 retains
    // exactly the final state after this chunk, without rollback snapshots.
    auto *packed = ggml_gated_delta_net(ctx, q, k, v, decay, beta, state, 1);
    auto *output = ggml_view_4d(ctx, packed, key_dim, value_heads, tokens, 1,
                                head_stride, size_t(inner) * scalar_bytes,
                                size_t(inner * tokens) * scalar_bytes, 0);
    auto *next_state = ggml_view_4d(ctx, packed, key_dim, key_dim, value_heads, 1,
                                    head_stride, size_t(key_dim * key_dim) * scalar_bytes,
                                    size_t(key_dim * key_dim * value_heads) * scalar_bytes,
                                    size_t(inner * tokens) * scalar_bytes);
    // Persist the returned state explicitly; the operator never mutates its
    // state input. This copy is ordered after all reads in the fused kernel.
    ggml_build_forward_expand(g.graph, ggml_cpy(ctx, next_state, state));

    auto *gated = ggml_mul(ctx, g.norm(output, l.ssm_norm), ggml_silu(ctx, z));
    auto *flat = ggml_reshape_2d(ctx, gated, inner, tokens);
    return g.mm(l.ssm_out, flat);
}

} // namespace strata::dense
