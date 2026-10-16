#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::deepseek_v4 {
// block_mxfp4 is 17 bytes. Read each unaligned four-byte word through
// naturally aligned words; the zero-shift case never reads past the word.
static __device__ __forceinline__ int mxfp4_word(const uint8_t* bytes, int word) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(bytes + 4 * word);
    const auto* aligned = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
    const unsigned offset = unsigned(address & 3);
    const uint32_t low = aligned[0];
    const uint32_t high = aligned[offset != 0];
    return int(__funnelshift_r(low, high, offset * 8));
}
// The canonical FP4 lookup, integer dot products, scale expression and FP32
// accumulation stay unchanged. This adapter changes only the packed load.
static __device__ __forceinline__ float mxfp4_dot_load(
        const void* __restrict__ weights, const block_q8_1* __restrict__ activation,
        const int& block, const int& index) {
    const auto* fp4 = static_cast<const block_mxfp4*>(weights) + block;
    const auto* q8 = reinterpret_cast<const int*>(activation->qs) + index;
    int sum = 0;
#pragma unroll
    for (int part = 0; part < VDR_MXFP4_Q8_1_MMVQ; ++part) {
        const int packed = mxfp4_word(fp4->qs, index + part);
        const int2 values = get_int_from_table_16(packed, kvalues_mxfp4);
        sum = ggml_cuda_dp4a(values.x, q8[part], sum);
        sum = ggml_cuda_dp4a(values.y, q8[part + 4], sum);
    }
    const float scale = ggml_cuda_e8m0_to_fp32(fp4->e) * 0.5f * __low2float(activation->ds);
    return scale * sum;
}
inline bool mxfp4_load_enabled(int cc, int columns, int rows, int tokens) {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_DSV4_MXFP4_LOAD");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled && cc == 1200 && (tokens == 1 || tokens == 5) &&
           ((columns == 4096 && rows == 2048) || (columns == 2048 && rows == 4096));
}
}
