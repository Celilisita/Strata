#pragma once
#include <cuda_runtime.h>
namespace strata::qwen35::native {
// Exact original NVFP4 -> FP16. Requires input4/output16 alignment;
// each lane owns eight consecutive values and emits one uint4 store.
void nvfp4_decode_raw_half8(const void*,void*,int rows,int columns,cudaStream_t);
}
