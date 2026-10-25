#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace strata::completed_v100::qwen35::native {
// Explicit precision experiment. Its FP32 activation quantization error is
// separate from the same-encoded-input operator mathematics gate.
void nvfp4_q8fs_prepare(const float*,void*,int columns,int tokens,cudaStream_t);
void nvfp4_q8fs_reconstruct(const void*,float*,int columns,int tokens,cudaStream_t);
void nvfp4_q8fs_project(const void* gate,const float* gate_scale,const void* up,const float* up_scale,
    const void* encoded,float* first,float* second,float* activation,int rows,int columns,int tokens,cudaStream_t);
}
