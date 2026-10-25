#pragma once
#include <cuda_runtime.h>

namespace strata::qwen35::native {
// Decode the original64-value/36-byte block into exact raw FP16 values.
// Per-tensor global scale remains an FP32 GEMM alpha/epilogue coefficient.
// A bounded caller-owned arena is reused; this does not requantize the model.
void nvfp4_decode_raw_half(const void* blocks,void* half_output,int rows,int columns,
                           cudaStream_t stream,float* diagnostic_values=nullptr);
}
