#pragma once
#include "attention_physical_pp.hpp"

namespace strata::gemma4::native::detail241 {
constexpr int QueryTile=PhysicalPpQueries,KeyTile=PhysicalPpKeys;
struct AttentionTile {float *scores,*partial,*acc,*maximum,*sum,*a,*b;};
struct PhysicalKv {float *k,*v01,*v23;int end;};
struct PointerTables {
    const float **key,**value,**query;
    float **score,**partial;
};
void unpack_physical(PhysicalKv,CacheView,cudaStream_t);
void prepare_pointers(PointerTables,PhysicalKv,AttentionTile,const float*,int,int,cudaStream_t);
void initialize_tile(AttentionTile,int,int,cudaStream_t);
void normalize_tile(AttentionTile,CacheView,const int*,int,int,int,int,cudaStream_t);
void accumulate_tile(AttentionTile,int,int,int,cudaStream_t);
void finish_tile(AttentionTile,float*,int,int,int,int,cudaStream_t);
}
