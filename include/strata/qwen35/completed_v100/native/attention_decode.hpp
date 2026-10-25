#pragma once
#include <cuda_runtime.h>
#include "strata/qwen35/completed_v100/native/decode_control.hpp"

namespace strata::completed_v100::qwen35::native {
struct DecodeAttentionShape {
    int dim, query_heads, kv_heads, valid, capacity;
    int kv_tile = 64;
    float scale = 1.0f;
    int partitions = 0; // Zero retains one contiguous tile per partition.
    bool q8=false;
};
// Explicit numeric partition policy; it does not prescribe physical occupancy.
// The calibrated resident capacity and actual SM count select online boundaries.
struct DecodeRounding { int processors=0,resident_blocks=2,key_granularity=256; };
__host__ __device__ inline int decode_partitions_unchecked(int valid,int heads,DecodeRounding r){
    const int bound=(valid+r.key_granularity-1)/r.key_granularity,wave=r.processors*r.resident_blocks;
    int best=r.resident_blocks<bound?r.resident_blocks:bound,fill=-1,best_waves=0;
    for(int parts=best;parts<=bound;++parts){
        const int blocks=heads*parts,waves=(blocks+wave-1)/wave,efficiency=100*blocks/(waves*wave);
        if(fill>=95&&waves>best_waves)break;
        if(efficiency>fill){best=parts;fill=efficiency;best_waves=waves;}
    }
    return best;
}
int decode_partition_count(int valid,int query_heads,DecodeRounding);
// Q[heads,256], Q4_0 K/V[capacity,kv_heads,8,18]; one query at valid-1.
// Caller owns the queried workspace through stream completion; no hot allocation.
size_t decode_attention_workspace(DecodeAttentionShape);
void decode_attention(const float *q, const void *k, const void *v, float *output,
                      void *workspace, size_t workspace_bytes, DecodeAttentionShape, cudaStream_t);
void decode_attention_graph(const float *,const void *,const void *,float *,void *,size_t,
                            DecodeAttentionShape,const DecodeCursor *,int maximum_partitions,cudaStream_t);
}
