#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::deepseek_v4::cuda_q8 {
struct Q80 { half d; int8_t qs[32]; };
struct Q81 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q80) == 34 && sizeof(Q81) == 36);
__device__ __forceinline__ int pack4(const int8_t * p) {
    return int(uint8_t(p[0])) | (int(uint8_t(p[1])) << 8) |
           (int(uint8_t(p[2])) << 16) | (int(uint8_t(p[3])) << 24);
}
__device__ __forceinline__ float warp_sum(float value) {
    for (int step = 16; step; step >>= 1) value += __shfl_down_sync(0xffffffffu, value, step);
    return value;
}
}
