#pragma once
#include "attention_swa.hpp"
#include "attention_blas_ops.hpp"
#include <cuda_fp16.h>

namespace strata::gemma4::native::swa403::detail403 {
struct Tile {
    native::detail::AttentionTile f32;
    half *query,*key,*value,*probability,*residual;
};
// Query/P/partial retain [Qhead][query128][dimension/key512]. Two consecutive
// query heads are two 128-column groups in one GEMM batch for their KV head.
// K/V are [KVhead16][key512][dimension256], never duplicated for GQA2.
Tile bind(void*,std::size_t);
void validate(const float*,float*,CacheView,const int*,int,int,int,Lease,Binding);
void pack_query(Tile,const float*,const int*,int past,int first,cudaStream_t);
void pack_kv(Tile,CacheView,int first,int keys,cudaStream_t);
void pack_probability(Tile,int keys,cudaStream_t);
}
