#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace strata::qwen35::native {
struct RmsConcatResources {int registers,blocks_per_sm,binary_version;size_t shared_bytes,local_bytes;};
RmsConcatResources rmsconcat_resources();
void rmsconcat(const float *input,const float *gamma,float *concat,int features,int rows,int offset,float epsilon,cudaStream_t);
}
