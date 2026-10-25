#pragma once
#include <cuda_runtime_api.h>
namespace strata::completed_v100::qwen35::native {
void prefill_down_tensor(const void*,const void*,float*,int,int,int,const float*,const float*,cudaStream_t);
}
