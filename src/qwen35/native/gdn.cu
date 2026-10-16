#include "strata/qwen35/native/gdn.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cmath>
#include "strata/qwen35/native/gdn_prefix.hpp"

namespace strata::qwen35::native {
namespace {
// Each warp owns two value rows and stripes key columns across its lanes.
// This keeps the key-contiguous state accesses coalesced and requires no CTA
// barriers. FP32 state stays in registers throughout the sequence.
template<int Keys, bool Retain = false>
__global__ void recurrence(GdnInputs x, GdnShape shape, float *output, float *final_state, float query_scale, GdnPrefix prefix = {}) {
    constexpr int Rows = 2, Columns = Keys / 32;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int first_value = int(blockIdx.x) * (4 * Rows) + warp * Rows, head = int(blockIdx.y);
    const int query_head = head % shape.query_heads;
    float state[Rows][Columns];
    for (int r = 0; r < Rows; ++r)
        for (int j = 0; j < Columns; ++j)
            state[r][j] = x.state[(size_t(head) * shape.value_dim + first_value + r) * Keys + lane + 32 * j];
    for (int t = 0; t < shape.tokens; ++t) {
        const size_t q_offset = (size_t(t) * shape.query_heads + query_head) * Keys;
        float decay = 0, beta = 0;
        if (!lane) {
            decay = __expf(x.log_decay[size_t(t) * shape.value_heads + head]);
            beta = x.beta[size_t(t) * shape.value_heads + head];
        }
        decay = __shfl_sync(0xffffffff, decay, 0);
        beta = __shfl_sync(0xffffffff, beta, 0);
        float q[Columns], k[Columns];
        for (int j = 0; j < Columns; ++j) {
            q[j] = x.q[q_offset + lane + 32 * j];
            k[j] = x.k[q_offset + lane + 32 * j];
        }
        for (int r = 0; r < Rows; ++r) {
            float key_memory = 0;
            for (int j = 0; j < Columns; ++j)
                key_memory = __fmaf_rn(state[r][j], k[j], key_memory);
            for (int d = 16; d; d /= 2)
                key_memory = __fadd_rn(key_memory, __shfl_xor_sync(0xffffffff, key_memory, d));
            const size_t index = (size_t(t) * shape.value_heads + head) * shape.value_dim + first_value + r;
            const float delta = __fmul_rn(__fmaf_rn(-decay, key_memory, x.v[index]), beta);
            float query_memory = 0;
            for (int j = 0; j < Columns; ++j) {
                state[r][j] = __fmaf_rn(decay, state[r][j], __fmul_rn(k[j], delta));
                if constexpr (Retain) if (t + 1 < shape.tokens)
                    prefix.state[t][(size_t(head) * shape.value_dim + first_value + r) * Keys + lane + 32 * j] = state[r][j];
                query_memory = __fmaf_rn(state[r][j], q[j], query_memory);
            }
            for (int d = 16; d; d /= 2)
                query_memory = __fadd_rn(query_memory, __shfl_xor_sync(0xffffffff, query_memory, d));
            if (!lane) output[index] = __fmul_rn(query_memory, query_scale);
        }
    }
    for (int r = 0; r < Rows; ++r)
        for (int j = 0; j < Columns; ++j)
            final_state[(size_t(head) * shape.value_dim + first_value + r) * Keys + lane + 32 * j] = state[r][j];
}
}
void gdn_recurrence(GdnInputs input, GdnShape shape, float *output, float *state, cudaStream_t stream) {
    if (!input.q || !input.k || !input.v || !input.log_decay || !input.beta || !input.state || !output || !state ||
        (shape.key_dim != 64 && shape.key_dim != 128) || shape.value_dim < 32 || shape.value_dim % 32 ||
        shape.value_dim > 256 || shape.query_heads < 1 || shape.value_heads % shape.query_heads ||
        shape.value_heads > 256 || shape.value_heads < 1 || shape.tokens < 1 || shape.tokens > 8192)
        throw std::invalid_argument("invalid native GDN geometry or buffers");
    const dim3 grid(shape.value_dim / 8, shape.value_heads);
    const float scale = 1.0f / std::sqrt(float(shape.key_dim));
    if (shape.key_dim == 128) recurrence<128><<<grid, 128, 0, stream>>>(input, shape, output, state, scale);
    else recurrence<64><<<grid, 128, 0, stream>>>(input, shape, output, state, scale);
    check_cuda(cudaGetLastError(), "native FP32 GDN recurrence");
}
void gdn_recurrence_prefix(GdnInputs input, GdnShape shape, float *output, float *state,
                           GdnPrefix prefix, cudaStream_t stream) {
    if (!input.q || !input.k || !input.v || !input.log_decay || !input.beta || !input.state ||
        !output || !state || !prefix.state[0] || (shape.tokens == 3 && !prefix.state[1]) ||
        shape.key_dim != 128 || shape.value_dim != 128 || shape.query_heads < 1 ||
        shape.value_heads < 1 || shape.value_heads > 256 || shape.value_heads % shape.query_heads ||
        (shape.tokens != 2 && shape.tokens != 3) || prefix.state[0] == input.state ||
        prefix.state[0] == state || (shape.tokens == 3 && (prefix.state[1] == input.state ||
        prefix.state[1] == state || prefix.state[1] == prefix.state[0])))
        throw std::invalid_argument("invalid retained GDN prefix lease");
    const dim3 grid(shape.value_dim / 8, shape.value_heads);
    recurrence<128, true><<<grid, 128, 0, stream>>>(input, shape, output, state,
                                                  1.0f / std::sqrt(float(shape.key_dim)), prefix);
    check_cuda(cudaGetLastError(), "native GDN retained prefix recurrence");
}

}
