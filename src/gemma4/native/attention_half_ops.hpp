#pragma once
#include "attention_blas_ops.hpp"
#include <cuda_fp16.h>

namespace strata::gemma4::native::detail {
// FP32 online-softmax state remains authoritative. These independent FP16
// operands are a request-scoped lease within the existing attention arena.
struct HalfAttentionTile {
    AttentionTile f32;
    half *q,*k,*v,*probabilities,*probability_residual;
};

size_t half_attention_tile_bytes(int heads,int dim);
HalfAttentionTile bind_half_attention_tile(void*,size_t,int heads,int dim);
// Q: [head][QueryTile][dim], K/V: [head][KeyTile][dim],
// Each probability plane: [head][QueryTile][KeyTile]. The residual plane
// represents (p - float(high))*4096. Invalid tile padding is not read.
void pack_half_query(HalfAttentionTile,const float*,int query_first,int queries,
    int heads,int dim,cudaStream_t);
void pack_half_kv(HalfAttentionTile,int keys,int heads,int dim,cudaStream_t);
void pack_half_probabilities(HalfAttentionTile,int keys,int queries,int heads,cudaStream_t);
}

namespace strata::gemma4::native {
void half_blas_attention(const float*,float*,CacheView,const int*,int past,
    int heads,int tokens,AttentionWorkspace,cudaStream_t);
}
