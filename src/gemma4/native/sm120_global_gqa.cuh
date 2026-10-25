#pragma once
#include "sm120_global_gqa.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
namespace strata::gemma4::native::detail_global_gqa {
constexpr int Dim=512,QueryHeads=32,KvHeads=4,Gqa=8,Threads=256,Partition=128,Tile=64;
constexpr int RowBytes=(Dim/32)*34,SharedBytes=2*Tile*RowBytes;
static_assert(RowBytes==544&&SharedBytes==69632,"Fixed raw Q8 staging, no F16 recode");
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float read_q8(const unsigned char* row,int d){
    const auto* block=row+(d/32)*34;
    return __half2float(*reinterpret_cast<const half*>(block))*float(reinterpret_cast<const signed char*>(block+2)[d%32]);
}
__global__ void segment(const float* query,float* partial,CacheView c,const int* position,int parts){
    extern __shared__ __align__(16) unsigned char staged[];
    auto* key_tile=staged;auto* value_tile=staged+Tile*RowBytes;
    const int tid=threadIdx.x,lane=tid%32,head=int(blockIdx.x)*Gqa+tid/32,part=blockIdx.y;
    const int pos=*position;float* out=partial+(std::size_t(head)*parts+part)*(Dim+2);
    if(pos<0||pos>=c.capacity){for(int i=lane;i<Dim+2;i+=32)out[i]=CUDART_NAN_F;return;}
    const int begin=part*Partition,end=min(pos,begin+Partition-1),kh=blockIdx.x;
    float q[Dim/32],accum[Dim/32]{};
    #pragma unroll
    for(int j=0;j<Dim/32;++j)q[j]=query[std::size_t(head)*Dim+lane+32*j];
    float top=-CUDART_INF_F,denominator=0;
    // Both 64-key tiles reuse the same raw-byte staging. All eight warps
    // consume the tile before any thread may replace it with the next tile.
    for(int base=begin;base<=end;base+=Tile){
        for(int i=tid;i<SharedBytes;i+=Threads){
            const int side=i/(Tile*RowBytes),at=i%(Tile*RowBytes),key=base+at/RowBytes,column=at%RowBytes;
            unsigned char value=0;
            if(key<=end){const auto* source=static_cast<const unsigned char*>(side?c.value:c.key);
                value=source[(std::size_t(key)*KvHeads+kh)*RowBytes+column];}
            staged[i]=value;
        }
        __syncthreads();
        const int count=min(Tile,end-base+1);
        for(int key=0;key<count;++key){
            const auto* kr=key_tile+key*RowBytes;const auto* vr=value_tile+key*RowBytes;float score=0;
            #pragma unroll
            for(int j=0;j<Dim/32;++j)score=fmaf(q[j],read_q8(kr,lane+32*j),score);
            score=sum(score);const float next=fmaxf(top,score),old=expf(top-next),current=expf(score-next);
            #pragma unroll
            for(int j=0;j<Dim/32;++j)accum[j]=fmaf(current,read_q8(vr,lane+32*j),old*accum[j]);
            denominator=fmaf(denominator,old,current);top=next;
        }
        __syncthreads();
    }
    if(!lane){out[0]=top;out[1]=denominator;}
    #pragma unroll
    for(int j=0;j<Dim/32;++j)out[2+lane+32*j]=accum[j];
}
}
