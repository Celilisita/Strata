#include "attention_half_ops.hpp"

namespace strata::gemma4::native::detail {
namespace {
__global__ void query_half(HalfAttentionTile tile,const float* input,
    int first,int queries,int heads,int dim){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=size_t(heads)*QueryTile*dim)return;
    const int d=int(i%dim),query=int((i/dim)%QueryTile),head=int(i/(QueryTile*dim));
    if(query<queries)tile.q[i]=__float2half_rn(input[(size_t(first+query)*heads+head)*dim+d]);
}

__global__ void kv_half(HalfAttentionTile tile,int keys,int heads,int dim){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=size_t(heads)*KeyTile*dim||int((i/dim)%KeyTile)>=keys)return;
    tile.k[i]=__float2half_rn(tile.f32.k[i]);
    tile.v[i]=__float2half_rn(tile.f32.v[i]);
}

__global__ void probabilities_half(HalfAttentionTile tile,int keys,int queries,int heads){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=size_t(heads)*QueryTile*KeyTile)return;
    const int key=int(i%KeyTile),query=int((i/KeyTile)%QueryTile);
    if(key<keys&&query<queries){
        const float p=tile.f32.scores[i];
        const half high=__float2half_rn(p);
        tile.probabilities[i]=high;
        tile.probability_residual[i]=__float2half_rn((p-__half2float(high))*4096.f);
    }
}
}

void pack_half_query(HalfAttentionTile tile,const float* input,int first,int queries,
    int heads,int dim,cudaStream_t stream){
    query_half<<<unsigned((size_t(heads)*QueryTile*dim+255)/256),256,0,stream>>>(
        tile,input,first,queries,heads,dim);
}

void pack_half_kv(HalfAttentionTile tile,int keys,int heads,int dim,cudaStream_t stream){
    kv_half<<<unsigned((size_t(heads)*KeyTile*dim+255)/256),256,0,stream>>>(
        tile,keys,heads,dim);
}

void pack_half_probabilities(HalfAttentionTile tile,int keys,int queries,int heads,cudaStream_t stream){
    probabilities_half<<<unsigned((size_t(heads)*QueryTile*KeyTile+255)/256),256,0,stream>>>(
        tile,keys,queries,heads);
}
}
