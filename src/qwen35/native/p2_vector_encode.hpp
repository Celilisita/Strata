#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
namespace strata::qwen35::native {
void q8p2_encode_vector(const float*,void*,size_t,cudaStream_t);
void swiglu_encode_vector(const float*,const float*,const float*,void*,int,int,cudaStream_t);
}
