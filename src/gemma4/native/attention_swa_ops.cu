#include "attention_swa_ops.hpp"
#include <math_constants.h>

namespace strata::gemma4::native::swa403::detail403 {
namespace {
__global__ void query_pack(Tile tile,const float* query,const int* position,int past,int first){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=std::size_t(Heads)*QueryTile*Dim)return;
    const int d=int(i%Dim),q=int((i/Dim)%QueryTile),head=int(i/(QueryTile*Dim));
    // Position is authoritative on GPU; an inconsistent host tile schedule
    // poisons the candidate instead of silently computing a different mask.
    const float value=*position==past?query[(std::size_t(first+q)*Heads+head)*Dim+d]:CUDART_NAN_F;
    tile.query[i]=__float2half_rn(value);
}
__global__ void kv_pack(Tile tile,CacheView cache,int first,int keys){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=std::size_t(KvHeads)*KeyTile*Dim)return;
    const int d=int(i%Dim),key=int((i/Dim)%KeyTile),head=int(i/(KeyTile*Dim));
    if(key>=keys)return;
    const int slot=(first+key)%cache.capacity;
    const std::size_t at=((std::size_t(slot)*KvHeads+head)*(Dim/32)+d/32)*34;
    const auto* k=static_cast<const unsigned char*>(cache.key)+at;
    const auto* v=static_cast<const unsigned char*>(cache.value)+at;
    const float kd=__half2float(*reinterpret_cast<const half*>(k));
    const float vd=__half2float(*reinterpret_cast<const half*>(v));
    // Same F32 decode then RN16 as174+195. The two GQA heads share this decode.
    tile.key[i]=__float2half_rn(kd*float(reinterpret_cast<const signed char*>(k+2)[d%32]));
    tile.value[i]=__float2half_rn(vd*float(reinterpret_cast<const signed char*>(v+2)[d%32]));
}
__global__ void probability_pack(Tile tile,int keys){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=std::size_t(Heads)*QueryTile*KeyTile||int(i%KeyTile)>=keys)return;
    const float p=tile.f32.scores[i];const half high=__float2half_rn(p);
    tile.probability[i]=high;
    tile.residual[i]=__float2half_rn((p-__half2float(high))*4096.f);
}
}
void pack_query(Tile tile,const float* query,const int* position,int past,int first,cudaStream_t s){
    query_pack<<<unsigned((std::size_t(Heads)*QueryTile*Dim+255)/256),256,0,s>>>(tile,query,position,past,first);
}
void pack_kv(Tile tile,CacheView c,int first,int keys,cudaStream_t s){
    kv_pack<<<unsigned((std::size_t(KvHeads)*KeyTile*Dim+255)/256),256,0,s>>>(tile,c,first,keys);
}
void pack_probability(Tile tile,int keys,cudaStream_t s){
    probability_pack<<<unsigned((std::size_t(Heads)*QueryTile*KeyTile+255)/256),256,0,s>>>(tile,keys);
}
}
