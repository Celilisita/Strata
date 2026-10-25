#pragma once
#include "strata/qwen35/native/nvfp4.hpp"

namespace strata::qwen35::native {
// Same resident FP4 values/scales as Tensor Core PP; block32 Q8 activations
// expose the memory-limited1..4-token service without padding to a GEMM tile.
constexpr size_t NvTinyActivationBytes=36;
void nvfp4_tiny_quantize(const float *,void *,int columns,int tokens,cudaStream_t);
void nvfp4_tiny_linear(Nvfp4View,const void *,float *,int tokens,cudaStream_t);
void nvfp4_tiny_pair(Nvfp4View,Nvfp4View,const void *,float *,float *,float *,int tokens,cudaStream_t);
}
