#pragma once
#include "strata/kernels/radix_sampler.hpp"
#include <cuda_runtime.h>
#include <cstdint>
namespace strata::qwen35::native {
bool mtp_gumbel_supported();
inline bool mtp_gumbel_params(const strata::kernels::SamplerParams &p){
    return !p.greedy&&p.temperature>0&&p.top_k>=1&&p.top_k<=64;
}
// Reuses the exact filtered distribution already produced by RadixSampler.
// Explicit top_k1..64 only; callers retain the complete-vocabulary legacy path otherwise.
void mtp_gumbel_pick(const strata::kernels::RadixSampler&,int vocabulary,
    const strata::kernels::SamplerParams&,const uint64_t*,unsigned,int*,unsigned*,cudaStream_t,int draws=1);
}
