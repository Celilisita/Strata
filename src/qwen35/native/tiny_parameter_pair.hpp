#pragma once
#include "strata/qwen35/native/quant.hpp"
#include <cuda_runtime.h>
namespace strata::qwen35::native {
struct TinyParameterPairResources {
    int registers=0,blocks=0;size_t shared_bytes=0,local_bytes=0;
};
// Qualified T3 Q4Pairs service. This is two independent linear projections;
// unlike tiny_batch_pair, it has no SwiGLU or third output.
void tiny_batch_parameter_pair(Matrix,Matrix,const void *,float *,float *,int,cudaStream_t);
TinyParameterPairResources tiny_batch_parameter_pair_resources();
}
