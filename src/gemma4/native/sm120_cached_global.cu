#include "sm120_cached_global.hpp"
#include "pp_tc_ops.hpp"
#include <math_constants.h>

namespace strata::gemma4::native::cached_global_detail {
namespace {
__global__ void decode_history(Buffers b, CacheView cache, const int* position,
    int tokens, int upper, int staged) {
    const auto i = std::size_t(blockIdx.x) * 256 + threadIdx.x;
    const auto count = std::size_t(4) * staged * 512;
    const int past = *position;
    const bool valid = past >= 0 && past <= cache.capacity - tokens && past <= upper - tokens;
    if (!i && !valid) atomicExch(b.invalid, 1u);
    if (i >= count) return;
    const int d = i % 512, key = (i / 512) % staged, head = i / (512 * staged);
    const auto to = (std::size_t(head) * b.keys + key) * 512 + d;
    half k = __float2half_rn(0.f), v = k;
    if (valid && key < past + tokens) {
        const int slot = key % cache.capacity;
        const auto offset = ((std::size_t(slot) * 4 + head) * 16 + d / 32) * 34;
        const auto* raw_k = static_cast<const unsigned char*>(cache.key) + offset;
        const auto* raw_v = static_cast<const unsigned char*>(cache.value) + offset;
        // Exact parent unpack: original half scale -> F32 product -> half RN.
        k = __float2half_rn(__half2float(*reinterpret_cast<const half*>(raw_k)) *
            float(reinterpret_cast<const signed char*>(raw_k + 2)[d % 32]));
        v = __float2half_rn(__half2float(*reinterpret_cast<const half*>(raw_v)) *
            float(reinterpret_cast<const signed char*>(raw_v + 2)[d % 32]));
    }
    b.key[to] = k;
    b.value[to] = v;
}
__global__ void cached_table(Buffers b, pp_tc_detail::Buffers tile, int tokens, int query, int key) {
    const int head = threadIdx.x;
    if (head >= 32) return;
    const int kh = head / 8;
    tile.kt[head] = b.key + (std::size_t(kh) * b.keys + key) * 512;
    tile.vt[head] = b.value + (std::size_t(kh) * b.keys + key) * 512;
    tile.qt[head] = tile.q + (std::size_t(head) * tokens + query) * 512;
    tile.pt[head] = tile.probability + std::size_t(head) * pp_tc_detail::Query * pp_tc_detail::Keys;
    tile.st[head] = tile.scores + std::size_t(head) * pp_tc_detail::Query * pp_tc_detail::Keys;
    tile.ot[head] = tile.partial + std::size_t(head) * pp_tc_detail::Query * 512;
}
__global__ void propagate_invalid(Buffers b, float* out, int tokens) {
    const auto i = std::size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < std::size_t(tokens) * 32 * 512 && *b.invalid) out[i] = CUDART_NAN_F;
}
}
void stage(Buffers b, CacheView cache, const int* position, int tokens, int upper, cudaStream_t stream) {
    const int staged = staged_key_bound(upper);
    primitive::check_cuda(cudaMemsetAsync(b.invalid, 0, sizeof(unsigned), stream),
        "Gemma cached global invalidate reset");
    decode_history<<<unsigned((std::size_t(4) * staged * 512 + 255) / 256), 256, 0, stream>>>
        (b, cache, position, tokens, upper, staged);
}
void pointers(Buffers b, pp_tc_detail::Buffers tile, int tokens, int query, int key, cudaStream_t stream) {
    cached_table<<<1, 32, 0, stream>>>(b, tile, tokens, query, key);
}
void propagate(Buffers b, float* out, int tokens, cudaStream_t stream) {
    propagate_invalid<<<unsigned((std::size_t(tokens) * 32 * 512 + 255) / 256), 256, 0, stream>>>
        (b, out, tokens);
}
}
