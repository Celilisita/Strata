#pragma once
#include "strata/qwen35/completed_v100/native/quant.hpp"
#include <cuda_runtime.h>
namespace strata::completed_v100::qwen35::native {
void tiny_meta_linear(Matrix,const void *,float *,int,cudaStream_t);
void tiny_meta_pair(Matrix,Matrix,const void *,float *,float *,float *,int,cudaStream_t);
}
