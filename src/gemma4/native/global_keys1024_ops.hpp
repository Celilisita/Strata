#pragma once
#include "pp_tc_attention.hpp"
#include <cuda_fp16.h>
namespace strata::gemma4::native::global_keys1024 {
bool pp_tc_requested();
std::size_t pp_tc_workspace_bytes(int maximum_tokens);
void pp_tc_attention(const float*,float*,CacheView,const int*,int past,int tokens,
                     int maximum_tokens,AttentionWorkspace,cudaStream_t);
}
namespace strata::gemma4::native::global_keys1024::pp_tc_detail {
#ifndef STRATA_GEMMA_PP_TC_QUERY
#define STRATA_GEMMA_PP_TC_QUERY 512
#endif
constexpr int Heads=32,Query=STRATA_GEMMA_PP_TC_QUERY,Keys=1024;
static_assert(Query==512,"GlobalKey1024 keeps Query512");
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
