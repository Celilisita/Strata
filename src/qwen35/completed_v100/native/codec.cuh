#pragma once
#include "strata/qwen35/completed_v100/native/quant.hpp"
#include <cuda_fp16.h>

namespace strata::completed_v100::qwen35::native {
struct Q4Coefficients { float scale, minimum; };
__device__ inline Q4Coefficients q4_coefficients(const unsigned char *p, int group,
                                               BlockLayout layout = BlockLayout::Gguf) {
    const int part = group & 3;
    int scale = group >= 4 ? ((p[12 + part] & 15) | ((p[4 + part] >> 6) << 4))
                                : (p[4 + part] & 63);
    int minimum = group >= 4 ? ((p[12 + part] >> 4) | ((p[8 + part] >> 6) << 4))
                                  : (p[8 + part] & 63);
    if (layout == BlockLayout::Q4Pairs) {
        const int offset = 4 + (group / 2) * 3;
        const int metadata = (int(p[offset]) | (int(p[offset + 1]) << 8) | (int(p[offset + 2]) << 16)) >> ((group & 1) * 12);
        scale = metadata & 63; minimum = (metadata >> 6) & 63;
    }
    return {__fmul_rn(__half2float(*reinterpret_cast<const half *>(p)), float(scale)),
            __fmul_rn(__half2float(*reinterpret_cast<const half *>(p + 2)), float(minimum))};
}
// Direct element address equations derived from the GGUF block schema. This is
// not a copied vector-dot/dequantization kernel or framework-specific layout.
__device__ inline float block_element(const unsigned char *p, Quant type, int j,
                                     BlockLayout layout = BlockLayout::Gguf) {
    if (type == Quant::Q4K) {
        const int group = j / 32;
        const auto coefficients = q4_coefficients(p, group, layout);
        const int code = (p[16 + (j / 64) * 32 + j % 32] >> ((group & 1) * 4)) & 15;
        // Explicit operations keep the scale/min rounding contract inspectable.
        return __fsub_rn(__fmul_rn(coefficients.scale, float(code)), coefficients.minimum);
    }
    const int half_block = j / 128;
    const int subgroup = (j % 128) / 32;
    const int lane = j % 32;
    const int low_index = half_block * 64 + (subgroup & 1) * 32 + lane;
    const int lo = (p[low_index] >> ((subgroup / 2) * 4)) & 15;
    const int hi = (p[128 + half_block * 32 + lane] >> (subgroup * 2)) & 3;
    const int code = (lo | (hi << 4)) - 32;
    const int scale = reinterpret_cast<const signed char *>(p + 192)[j / 16];
    const float d = __half2float(*reinterpret_cast<const half *>(p + 208));
    return __fmul_rn(__fmul_rn(d, float(scale)), float(code));
}
__device__ inline half matrix_element(Matrix matrix, int row, int column) {
    const int stride = matrix.quant == Quant::Q4K ? 144 : 210;
    const size_t block = size_t(row) * (matrix.columns / 256) + column / 256;
    const auto *p = static_cast<const unsigned char *>(matrix.data) + block * stride;
    return __float2half_rn(block_element(p, matrix.quant, column % 256, matrix.layout));
}
// The caller assigns a warp to 32 contiguous columns of one row. Metadata is
// uniform over a Q4 group, and over each sixteen-column half of a Q6 group.
__device__ inline half warp_matrix_element(Matrix m, int row, int column) {
    const int lane = threadIdx.x % 32;
    const int j = column % 256;
    const int stride = m.quant == Quant::Q4K ? 144 : 210;
    const size_t block = size_t(row) * (m.columns / 256) + column / 256;
    const auto *p = static_cast<const unsigned char *>(m.data) + block * stride;
    if (m.quant == Quant::Q4K) {
        const int group = j / 32;
        const auto own = lane == 0 ? q4_coefficients(p, group, m.layout) : Q4Coefficients{0, 0};
        const float scale = __shfl_sync(0xffffffff, own.scale, 0);
        const float minimum = __shfl_sync(0xffffffff, own.minimum, 0);
        const int code = (p[16 + (j / 64) * 32 + j % 32] >> ((group & 1) * 4)) & 15;
        return __float2half_rn(__fsub_rn(__fmul_rn(scale, float(code)), minimum));
    }
    const int section = j / 128, group = (j % 128) / 32;
    float own = 0;
    if (lane % 16 == 0)
        own = __fmul_rn(__half2float(*reinterpret_cast<const half *>(p + 208)),
                        float(reinterpret_cast<const signed char *>(p + 192)[j / 16]));
    const float scale = __shfl_sync(0xffffffff, own, lane & 16);
    const int low = (p[section * 64 + (group & 1) * 32 + lane] >> ((group / 2) * 4)) & 15;
    const int high = (p[128 + section * 32 + lane] >> (group * 2)) & 3;
    return __float2half_rn(__fmul_rn(scale, float((low | (high << 4)) - 32)));
}
} // namespace strata::completed_v100::qwen35::native
