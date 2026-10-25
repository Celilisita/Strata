#pragma once
#include "ggml.h"
#include "ggml-strata-kda-precision.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {
// Process-immutable graph policy. Never change it while a model/context can
// reuse graphs. The explicit builder below does not read an environment flag.
inline bool kda_chunk_prefix_requested() {
    static const bool requested = [] {
        const char * value = std::getenv("STRATA_GLM53_KDA_CHUNK_PREFIX");
        return value && std::strcmp(value, "1") == 0;
    }();
    return requested;
}

inline bool kda_chunk_shape(const ggml_tensor * t, int64_t a, int64_t b, int64_t c, int64_t d) {
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        t->ne[0] == a && t->ne[1] == b && t->ne[2] == c && t->ne[3] == d;
}

// A single fixed PP domain. Other T/K/head/sequence/type/stride combinations,
// including TG and T<K, retain the original complete recurrent implementation.
inline bool kda_chunk_prefix_eligible(int64_t snapshots, ggml_tensor * q, ggml_tensor * k,
        ggml_tensor * v, ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state) {
    return snapshots == 3 && kda_chunk_shape(q, 128, 64, 8192, 1) &&
        kda_chunk_shape(k, 128, 64, 8192, 1) && kda_chunk_shape(v, 128, 64, 8192, 1) &&
        kda_chunk_shape(g, 128, 64, 8192, 1) && kda_chunk_shape(beta, 1, 64, 8192, 1) &&
        kda_chunk_shape(state, 128, 128, 64, 1);
}

struct KdaChunkPrefixInfo {
    int64_t prefix_tokens = 8189, tail_tokens = 3, window_tokens = 128;
    int64_t windows = 0, internal_chunks = 0, padded_positions = 0;
    int graph_nodes_added = 0, marked_matmul_nodes = 0;
    ggml_tensor * assembly_seed = nullptr;
    size_t assembly_seed_input_bytes = 0, assembly_seed_write_bytes = 0;
};

inline bool kda_chunk_output_shape(const ggml_tensor * t, int64_t count) {
    return t && t->type == GGML_TYPE_F32 && t->ne[0] == 128 &&
        t->ne[1] == 64 && t->ne[2] == count && t->ne[3] == 1;
}

inline ggml_tensor * kda_chunk_time_view(ggml_context * ctx, ggml_tensor * t, int64_t first, int64_t count) {
    return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], count, 1,
        t->nb[1], t->nb[2], size_t(count) * t->nb[2], size_t(first) * t->nb[2]);
}

// Chunk must call the existing vector-aware build_delta_net_chunking unchanged
// and return {windowY, finalState}. Tail must call the original recurrent/cache
// path for exactly three tokens; it expands GDN -> snapshot CPY before return.
// No copied recurrence or CUDA kernel is introduced by this window orchestration.
template<class Chunk, class Tail>
ggml_tensor * build_kda_chunk_prefix(ggml_context * ctx, ggml_cgraph * graph,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * g,
        ggml_tensor * beta, ggml_tensor * state, Chunk && chunk, Tail && tail,
        KdaChunkPrefixInfo * info = nullptr) {
    if (!ctx || !graph || !kda_chunk_prefix_eligible(3, q, k, v, g, beta, state))
        throw std::runtime_error("KDA chunk-prefix requires the fixed F32/S128/H64/T8192/N1/K3 domain");
    const int before = ggml_graph_n_nodes(graph);
    KdaChunkPrefixInfo observed;
    std::array<ggml_tensor *, 5> inputs{q, k, v, g, beta};
    // A naked NONE tensor would be an upfront-allocated graph leaf for every
    // layer. Seed fullY with a real PAD node depending on first-window output.
    // Its 256MiB write copies 4MiB of real output and zero-fills 252MiB; later
    // SETs overwrite that padding. This extra initialization is paid work.
    // SET chains then order all windows without repeated fullY concatenation.
    // Neither the assembly nor intermediate states are marked OUTPUT.
    ggml_tensor * output = nullptr;
    ggml_tensor * carry = state;
    for (int64_t first = 0; first < observed.prefix_tokens; first += observed.window_tokens) {
        const int64_t count = (std::min)(observed.window_tokens, observed.prefix_tokens - first);
        std::array<ggml_tensor *, 5> part{};
        for (size_t i = 0; i < part.size(); ++i) part[i] = kda_chunk_time_view(ctx, inputs[i], first, count);
        const int window_before = ggml_graph_n_nodes(graph);
        auto result = chunk(part[0], part[1], part[2], part[3], part[4], carry);
        if (!kda_chunk_output_shape(result.first, count) || !kda_chunk_shape(result.second, 128, 128, 64, 1))
            throw std::runtime_error("KDA chunk-prefix carry differs from original state geometry");
        ggml_tensor * window_output = ggml_cont(ctx, result.first);
        if (!output) {
            output = ggml_pad(ctx, window_output, 0, 0, 8192 - int(count), 0);
            observed.assembly_seed = output;
            observed.assembly_seed_input_bytes = ggml_nbytes(window_output);
            observed.assembly_seed_write_bytes = ggml_nbytes(output);
        } else {
            output = ggml_set_inplace(ctx, output, window_output,
                output->nb[1], output->nb[2], output->nb[3], size_t(first) * output->nb[2]);
        }
        carry = result.second;
        // Expand both branches now. The next window must not retain all prior
        // decay/triangular/output temporaries until a final aggregate expand.
        ggml_build_forward_expand(graph, output);
        ggml_build_forward_expand(graph, carry);
        // Only nodes introduced by this explicit candidate window are tagged.
        // The original shared formula and recurrent tail remain unchanged.
        for (int i = window_before; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT) {
                ggml_strata_kda_mark_f32(node);
                ++observed.marked_matmul_nodes;
            }
        }
        ++observed.windows;
        observed.internal_chunks += (count + 15) / 16;
        observed.padded_positions += (16 - count % 16) % 16;
    }
    std::array<ggml_tensor *, 5> last{};
    for (size_t i = 0; i < last.size(); ++i)
        last[i] = kda_chunk_time_view(ctx, inputs[i], observed.prefix_tokens, observed.tail_tokens);
    // The old tail path consumes the true nonzero prefix state. It writes the
    // original newest/one-back/two-back cache groups before output assembly.
    ggml_tensor * tail_output = tail(last[0], last[1], last[2], last[3], last[4], carry);
    if (!kda_chunk_output_shape(tail_output, observed.tail_tokens))
        throw std::runtime_error("KDA chunk-prefix tail output geometry differs");
    output = ggml_set_inplace(ctx, output, ggml_cont(ctx, tail_output),
        output->nb[1], output->nb[2], output->nb[3], size_t(observed.prefix_tokens) * output->nb[2]);
    ggml_build_forward_expand(graph, output);
    observed.graph_nodes_added = ggml_graph_n_nodes(graph) - before;
    if (info) *info = observed;
    return output;
}
} // namespace strata::glm5_next
