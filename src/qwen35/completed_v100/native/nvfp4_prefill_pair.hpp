#pragma once
#include <cuda_runtime.h>
namespace strata::completed_v100::qwen35::native {
void nvfp4_pair_swiglu(const float*,const float*,const float*,float*,int rows,int tokens,cudaStream_t);
}
