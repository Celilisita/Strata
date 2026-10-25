#pragma once
#include "fp4_compensated.hpp"
#include <array>
#include <cstdint>
namespace strata::qwen35::native {
inline std::array<uint64_t,3> &tiny_fused_submissions(){static thread_local std::array<uint64_t,3> calls{};return calls;}
CompensatedFp4Input fp4_tiny_fused_output(CompensatedFp4Input,int columns);
PackedFp4Resources fp4_tiny_fused_resources();
void fp4_tiny_fused_decoder(float*,cudaStream_t);
void fp4_tiny_fused_pair(PackedFp4,PackedFp4,CompensatedFp4Input,float,int,cudaStream_t,float *gate=nullptr,float *up=nullptr,float *activation=nullptr);
void fp4_tiny_fused_compare(CompensatedFp4Input,CompensatedFp4Input,int,int,unsigned*,cudaStream_t);
}
