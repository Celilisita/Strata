#pragma once
#include "attention_gqa_t2.hpp"
#include <cuda_fp16.h>
namespace strata::gemma4::native::gqa371::detail {
struct Tile {
    half *key,*value,*query,*probability;
    float *score,*partial,*split_partial,*maximum,*denominator;
    unsigned* query_bad;int* first_bad_key;
    int capacity,parts;
};
void validate(const float*,float*,CacheView,const int*,int,int,AttentionWorkspace,Lease,Binding,const unsigned*,const Layout&);
void initialize(Tile,cudaStream_t);
void stage_kv(Tile,CacheView,const int*,const unsigned*,cudaStream_t);
void pack_query(Tile,const float*,const int*,const unsigned*,cudaStream_t);
void softmax(Tile,const int*,const unsigned*,cudaStream_t);
void finish(Tile,float*,const int*,const unsigned*,cudaStream_t);
}
