#pragma once
#include <cuda_runtime.h>
namespace strata::qwen35::native {
// The encoded input is the existing056 AoS forT1/2 or063 SoA forT3.
// Only raw-Q8 load width changes; no weight/activation precision changes.
void raw_q8_wide_project(const void* raw,const void* encoded,float* output,
    int rows,int columns,int tokens,cudaStream_t);
}
