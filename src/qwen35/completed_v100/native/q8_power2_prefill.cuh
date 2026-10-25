#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
namespace strata::completed_v100::qwen35::native {
// Explicit precision option: group16 Q8 with power-of-two scale. Every encoded
// value is exactly FP16; original-F32 quantization error is a separate measure.
void q8p2_encode(const float*,void* half_output,size_t count,cudaStream_t);
void q8p2_reconstruct(const void* half_input,float*,size_t count,cudaStream_t);
struct Q8P2Workspace {
    void* half_weights;void* half_input;cublasHandle_t handle;
    void* vendor;size_t vendor_bytes;const float* zero;
    size_t weight_capacity,input_capacity;bool decode8=false;bool vector_encode=false;
};
void q8p2_nvfp4_prefill(const void* raw,const float* scale,const float* input,float* output,
    int rows,int columns,int tokens,Q8P2Workspace,cudaStream_t);
}
