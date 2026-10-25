#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace strata::qwen35::native {
// Three signed-byte residuals per activation, with independent 16-value scales.
// Original NVFP4 payload is read directly; no persistent weight conversion.
size_t nvfp4_packed_input_bytes(int width,int tokens);
void nvfp4_packed_projection(const void* blocks,const float* global_scale,const float* input,
    float* output,int rows,int columns,int tokens,void* input_workspace,cudaStream_t);
void nvfp4_packed_cached_projection(const void* blocks,const float* global_scale,const float* input,
    float* output,int rows,int columns,int tokens,void* input_workspace,cudaStream_t);

}
