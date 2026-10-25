#pragma once
#include "parent_ops.hpp"
#include <cuda_fp16.h>
namespace strata::gemma4::native::local_gqa2_candidate::pp_tc_detail {
#ifndef STRATA_GEMMA_PP_TC_QUERY
#define STRATA_GEMMA_PP_TC_QUERY 128
#endif
#ifndef STRATA_GEMMA_PP_TC_LOCAL_GQA2
#define STRATA_GEMMA_PP_TC_LOCAL_GQA2 0
#endif
static_assert(STRATA_GEMMA_PP_TC_LOCAL_GQA2==0||STRATA_GEMMA_PP_TC_LOCAL_GQA2==1,"local GQA2 requires0or1");
constexpr bool LocalGroupedGqa2=STRATA_GEMMA_PP_TC_LOCAL_GQA2==1;
constexpr int Heads=32,Query=STRATA_GEMMA_PP_TC_QUERY,Keys=512;
static_assert(!LocalGroupedGqa2||Query==512,"local GQA2 frozen Query512");
static_assert(Query==128||Query==256||Query==512,"Gemma PP query tile must be128/256/512");
using Buffers=::strata::gemma4::native::pp_tc_detail::Buffers;
using ::strata::gemma4::native::pp_tc_detail::bind;
void prepare(Buffers,const float*,float*,int,int,bool,cudaStream_t);
void decode(Buffers,CacheView,int first,int count,cudaStream_t);
void pointers(Buffers,CacheView,int tokens,int first,bool,cudaStream_t);
void normalize(Buffers,CacheView,const int*,int query_first,int key_first,int keys,int queries,bool,cudaStream_t);
void accumulate(Buffers,float*,int first,int queries,int dim,bool,cudaStream_t);
void finish(Buffers,float*,int tokens,int dim,cudaStream_t);
}
