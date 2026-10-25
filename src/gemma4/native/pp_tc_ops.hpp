#pragma once
#include "pp_tc_attention.hpp"
#include <cuda_fp16.h>
namespace strata::gemma4::native::pp_tc_detail {
#ifndef STRATA_GEMMA_PP_TC_QUERY
#define STRATA_GEMMA_PP_TC_QUERY 128
#endif
constexpr int Heads=32,Query=STRATA_GEMMA_PP_TC_QUERY,Keys=512;
static_assert(Query==128||Query==256||Query==512,"Gemma PP query tile must be128/256/512");
struct Buffers {
    half *q,*k,*v,*probability;
    float *scores,*partial,*maximum,*sum,*a,*b;
    const void **kt,**vt,**qt,**pt;
    void **st,**ot;
};
Buffers bind(void*,std::size_t,int maximum_tokens);
void prepare(Buffers,const float*,float*,int,int,cudaStream_t);
void decode(Buffers,CacheView,int first,int count,cudaStream_t);
void pointers(Buffers,CacheView,int tokens,int first,cudaStream_t);
void normalize(Buffers,CacheView,const int*,int query_first,int key_first,int keys,int queries,cudaStream_t);
void accumulate(Buffers,float*,int first,int queries,int dim,cudaStream_t);
void finish(Buffers,float*,int tokens,int dim,cudaStream_t);
}
