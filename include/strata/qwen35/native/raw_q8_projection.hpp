#pragma once
#include "strata/qwen35/native/original_formats.hpp"
#include <cuda_runtime.h>
namespace strata::qwen35::native {
// Raw F32 input; original GGUF Q8 weights, including vocabulary and MTP roles.
void raw_q8_projection(Matrix,const float*,float*,int tokens,cudaStream_t);
void raw_q8_projection_warp(Matrix,const float* input,float* output,int tokens,int warps_per_row,cudaStream_t);
void raw_q8_packed_projection(Matrix,const float*,float*,int tokens,int group,void* workspace,cudaStream_t);
void raw_q8_blocked_projection(Matrix,const float*,float*,int tokens,int group,void* workspace,cudaStream_t);
}
