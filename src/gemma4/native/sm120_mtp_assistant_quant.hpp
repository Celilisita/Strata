#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
namespace strata::gemma4::native::mtp_assistant {
struct Q40Matrix {const void* data=nullptr;int rows=0,columns=0;};
// GGUF Q4_0: fp16 signed scale +16 packed bytes, 32 values/block.
// Decoded output is F32 [row_count][columns], first_row is a matrix row.
void q40_decode_rows(Q40Matrix,float* decoded,int first_row,int row_count,cudaStream_t);
// Full one-token projection; all row chunks and decoding are part of this call.
// decoded_values counts floats. The head uses eight32768-row chunks (128MiB).
void q40_project(Q40Matrix,const float* input,float* output,float* decoded,size_t decoded_values,
    cublasHandle_t,void* blas_workspace,size_t workspace_bytes,cudaStream_t,int row_chunk=32768);
// Explicit Gemma condition: scaled MAIN embedding followed by final-RMS seed.
void condition_concat(const float*,const float*,float*,unsigned* invalid,cudaStream_t);
}
