#pragma once
#include <cuda_runtime.h>

namespace strata::completed_v100::qwen35::native {
// GPU codec supplies bounded dense tiles. Matrix multiplication, global/shared
// iterators, MMA and epilogue are the NVIDIA CUTLASS implementation.
void original_cutlass_half(const void* weights,const void* input,float* output,
    int rows,int columns,int tokens,int output_stride,
    const float* alpha,const float* beta,cudaStream_t);
void original_cutlass_float(const float* weights,const float* input,float* output,
    int rows,int columns,int tokens,int output_stride,
    const float* alpha,const float* beta,cudaStream_t);
}
