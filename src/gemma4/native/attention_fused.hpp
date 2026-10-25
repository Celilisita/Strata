#pragma once
#include "strata/gemma4/native/attention_native.hpp"

namespace strata::gemma4::native {
// Direct primitive candidate only: N1280, past0, local D256 or global D512.
// Q/output stay [token][32][dim]. Arena contains duplicated F32 K/V and,
// only for SWA, one [1280][1280] F32 bias shared by all heads.
size_t fused_float_attention_bytes(CacheView,int heads,int tokens);
size_t fused_float_attention_shared_bytes();
void fused_float_attention(const float*,float*,CacheView,const int* position,
    int past,int heads,int tokens,AttentionWorkspace,cudaStream_t);
}
