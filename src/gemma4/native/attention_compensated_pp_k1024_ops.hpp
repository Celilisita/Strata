#pragma once
#include "attention_compensated_pp_k1024.hpp"
#include <cuda_fp16.h>

namespace strata::gemma4::native::detail324 {
constexpr int QueryTile=CompensatedPpK1024Queries,KeyTile=CompensatedPpK1024Keys;
constexpr int Heads=32,KvHeads=4,Dim=512,HeadsPerKv=8;
constexpr std::size_t TileElements=std::size_t(Heads)*QueryTile*Dim;
constexpr std::size_t ScoreElements=std::size_t(Heads)*QueryTile*KeyTile;
struct AttentionTile {
    half *q,*p;
    float *work,*acc,*maximum,*sum,*a,*b;
};
struct PhysicalKv {half *k,*v01,*v23;int end;};
struct PointerTables {
    const void **key[2],**value[2],**query[2],**probability[2];
    void **score_work,**partial_work;
};
void unpack_physical(PhysicalKv,CacheView,cudaStream_t);
void pack_query(AttentionTile,const float*,int,cudaStream_t);
void prepare_pointers(PointerTables,PhysicalKv,AttentionTile,int,cudaStream_t);
void initialize_tile(AttentionTile,cudaStream_t);
void normalize_tile(AttentionTile,const int*,int,int,int,int,cudaStream_t);
void accumulate_tile(AttentionTile,cudaStream_t);
void finish_tile(AttentionTile,float*,int,cudaStream_t);
}
