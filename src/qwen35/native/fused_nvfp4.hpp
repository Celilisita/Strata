#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
namespace strata::qwen35::native {
// Optional FP32 diagnostic activation; normal execution writes only FP4/SF.
void nvfp4_fused_swiglu(const float *gate,const float *up,float *diagnostic,
    Nvfp4Input,int tokens,int columns,float global,cudaStream_t);
void nvfp4_fused_swiglu_half(const void *half_gate,const void *half_up,float *diagnostic,
    Nvfp4Input,int tokens,int columns,float global,cudaStream_t);
void nvfp4_expand_half(const void *,float *,size_t,cudaStream_t);
}
