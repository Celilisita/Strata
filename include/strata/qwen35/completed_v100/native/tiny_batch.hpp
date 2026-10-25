#pragma once
#include "strata/qwen35/completed_v100/native/quant.hpp"
#include <cuda_runtime.h>
namespace strata::completed_v100::qwen35::native {
// Two/three token rows; same signed activation format and FP32 vector math.
void tiny_batch_quantize(const float *,void *,int columns,int tokens,cudaStream_t);
void tiny_batch_linear(Matrix,const void *,float *,int tokens,cudaStream_t);
void tiny_batch_pair(Matrix,Matrix,const void *,float *,float *,float *,int tokens,cudaStream_t);
}
