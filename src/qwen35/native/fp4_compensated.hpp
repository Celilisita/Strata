#pragma once
#include "fp4_packed.hpp"
namespace strata::qwen35::native {
constexpr float CompensatedLowRatio=1.f/16.f;
struct CompensatedFp4Input {Nvfp4Input high,low;};
void fp4_compensated_quantize(const float*,CompensatedFp4Input,int tokens,int columns,float global,cudaStream_t);
PackedFp4Resources fp4_compensated_resources(bool pair);
void fp4_compensated_linear(PackedFp4,CompensatedFp4Input,float*,int,cudaStream_t);
void fp4_compensated_pair(PackedFp4,PackedFp4,CompensatedFp4Input,float*,float*,float*,int,cudaStream_t);
}
