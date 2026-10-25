#pragma once
#include "strata/kernels/radix_sampler.hpp"
namespace strata::completed_v100::qwen35::native {
// Exact frozen positioned draw: counter=base+GPUposition-origin+row.
void sample_positioned(strata::kernels::RadixSampler&,const float*,const int*,int,
    const strata::kernels::SamplerParams&,const int*,int,int,void*,const int* = nullptr);
}
