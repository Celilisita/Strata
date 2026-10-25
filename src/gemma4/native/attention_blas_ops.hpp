#pragma once
#include "strata/gemma4/native/attention_native.hpp"
namespace strata::gemma4::native::detail {
constexpr int QueryTile=128,KeyTile=512;
struct AttentionTile {
    float *k,*v,*scores,*partial,*acc,*maximum,*sum,*a,*b;
};
void initialize_tile(AttentionTile,int,int,cudaStream_t);
void unpack_tile(AttentionTile,CacheView,int,int,int,cudaStream_t);
void normalize_tile(AttentionTile,CacheView,const int*,int,int,int,int,cudaStream_t);
void accumulate_tile(AttentionTile,int,int,int,cudaStream_t);
void finish_tile(AttentionTile,float*,int,int,int,int,cudaStream_t);
}
