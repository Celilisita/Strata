#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace strata::qwen35::native {
// Experimental native Tensor Core primitive. Q8 group16 semantics are the same
// as the admitted single-Q8 path; FP16 high/low encoding approximates its F32
// reconstruction to the same-input numerical gate. No weight requantization.
void nvfp4_m8_prepare(const float*,void*,int columns,int tokens,cudaStream_t);
void nvfp4_m8_project(const void* gate,const float* gate_scale,const void* up,const float* up_scale,
    const void* encoded,float* first,float* second,float* activation,int rows,int columns,int tokens,
    int splits,float* partial,size_t partial_values,cudaStream_t);
}
