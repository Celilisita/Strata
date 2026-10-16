#pragma once
#include <cuda_runtime.h>

namespace strata::qwen35::native {
// Text positions; contiguous [tokens,heads,dim], partial NEOX rotary pairs.
void rotary_text(const float *, const int *positions, float *, int dim, int rotary_dim,
                 int heads, int tokens, float frequency_base, cudaStream_t);
void rotary_visual(const float *, const int *positions, float *, int dim, int rotary_dim,
                   int heads, int tokens, float frequency_base, cudaStream_t, const int* table);
// Normalized power-of-two butterflies; output may alias input.
void hadamard(const float *, float *, int width, size_t rows, cudaStream_t);
// Q4_0:18 bytes per32 values. Cache [capacity,heads,dim/32,18], token-major.
// Input is contiguous [tokens,heads,dim]; writes exactly [start,start+tokens).
void q4_cache_write(const float *, void *, int dim, int heads, int tokens,
                    int start, int capacity, cudaStream_t);
void q4_cache_write_position(const float *,void *,int dim,int heads,const int *position,cudaStream_t);
// Extract the second half of each interleaved Q/gate row and apply sigmoid.
void attention_gate(const float *q_gate, const float *attention, float *raw_gate,
                    float *sigmoid, float *gated, int dim, int heads, int tokens, cudaStream_t);
}
