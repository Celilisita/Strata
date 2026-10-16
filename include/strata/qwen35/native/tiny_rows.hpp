#pragma once
#include "strata/qwen35/native/quant.hpp"
#include <cuda_runtime.h>
namespace strata::qwen35::native {
void tiny_row_linear(Matrix,const void *,float *,int,cudaStream_t);
void tiny_row_pair(Matrix,Matrix,const void *,float *,float *,float *,int,cudaStream_t);
}
