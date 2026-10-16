#pragma once
#include "strata/kernels/sampler.hpp"
#include <cuda_runtime.h>

namespace strata::kernels::sampler_math {
__device__ __forceinline__ float uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = uint32_t(counter), c1 = uint32_t(counter >> 32);
    uint32_t c2 = uint32_t(seed), c3 = uint32_t(seed >> 32);
    for (uint32_t round = 0; round < 10; ++round) {
        const uint32_t a = __umulhi(0xBB67AE85u, c2) ^ c1 ^ round;
        const uint32_t b = 0xBB67AE85u * c2;
        const uint32_t c = __umulhi(0x9E3779B9u, c0) ^ c3;
        const uint32_t d = 0x9E3779B9u * c0;
        c0 = a; c1 = b; c2 = c; c3 = d;
    }
    return float(c0 >> 8) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float penalize(float value, int count, const SamplerParams& p) {
    if (count <= 0) return value;
    value = value <= 0.0f ? value * p.penalty_repeat : value / p.penalty_repeat;
    return value - (float(count) * p.penalty_freq + p.penalty_present);
}
} // namespace strata::kernels::sampler_math
