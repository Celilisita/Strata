#pragma once
#include "strata/gemma4/native/decoder_ops.hpp"
namespace strata::gemma4::native {
// Direct packed Q8 reads; fixed Graph grid covers reserved actual KV capacity.
// Every replay reads its absolute query position from device state.
void split_attention(const float*,float*,CacheView,const int*,int,int,AttentionWorkspace,cudaStream_t);
// FP32 cuBLAS query/KV tiles, bounded scratch, exact causal/SWA masks.
void blas_attention(const float*,float*,CacheView,const int*,int,int,int,AttentionWorkspace,cudaStream_t);
}
