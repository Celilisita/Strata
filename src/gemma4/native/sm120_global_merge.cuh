#pragma once
#include "sm120_global_merge.hpp"
// Adapted only from the V100266 merge; original segment remains the sole producer.
namespace strata::gemma4::native::detail_global_merge {
constexpr int Dim=512,Heads=32,Threads=128,Tile=128,Warps=4,MaxParts=2048;
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float maximum(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void merge(const float* partial,float* output,const int* position,int capacity,int max_parts){
    __shared__ float weights[MaxParts],reduction[Warps],vectors[Warps*Tile],top,denominator;
    __shared__ unsigned char present[MaxParts];
    __shared__ unsigned invalid[Warps],bad_stats;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.x,tile=blockIdx.y,p=*position;
    const size_t out=size_t(head)*Dim+tile*Tile+tid;
    if(p<0||p>=capacity){output[out]=CUDART_NAN_F;return;}
    const int parts=(p+128)/128;const size_t base=size_t(head)*max_parts;
    float value=-CUDART_INF_F;bool bad=false;
    for(int part=tid;part<parts;part+=Threads){
        const float z=partial[(base+part)*(Dim+2)+1],m=partial[(base+part)*(Dim+2)];
        bad|=!isfinite(z)||z<0||(z>0&&!isfinite(m));if(z>0)value=fmaxf(value,m);
    }
    const unsigned mask=__ballot_sync(0xffffffff,bad);value=maximum(value);
    if(!lane){invalid[warp]=mask;reduction[warp]=value;}__syncthreads();
    if(!warp){const float all=maximum(lane<Warps?reduction[lane]:-CUDART_INF_F);
        if(!lane){top=all;bad_stats=invalid[0]|invalid[1]|invalid[2]|invalid[3];}}
    __syncthreads();float total=0;
    for(int part=tid;part<parts;part+=Threads){
        const float z=partial[(base+part)*(Dim+2)+1];const bool active=z>0;
        const float weight=active?expf(partial[(base+part)*(Dim+2)]-top):0.f;
        present[part]=active;weights[part]=weight;total=fmaf(z,weight,total);
    }
    total=sum(total);if(!lane)reduction[warp]=total;__syncthreads();
    if(!warp){const float all=sum(lane<Warps?reduction[lane]:0.f);if(!lane)denominator=all;}
    __syncthreads();
    // Four part stripes directly consume the original AoS, without a conversion.
    // z>0 retains active NaN numerators even when their exp weight underflows.
    float accum[Tile/32]{};
    for(int part=warp;part<parts;part+=Warps)if(present[part]){
        const float weight=weights[part];
        #pragma unroll
        for(int j=0;j<Tile/32;++j)
            accum[j]=fmaf(weight,partial[(base+part)*(Dim+2)+2+tile*Tile+lane+j*32],accum[j]);
    }
    #pragma unroll
    for(int j=0;j<Tile/32;++j)vectors[warp*Tile+lane+j*32]=accum[j];
    __syncthreads();value=0;
    #pragma unroll
    for(int w=0;w<Warps;++w)value=__fadd_rn(value,vectors[w*Tile+tid]);
    output[out]=bad_stats||!(denominator>0)||!isfinite(value)?CUDART_NAN_F:value/denominator;
}
inline int parts(int capacity,AttentionWorkspace w){
    if(capacity<1||capacity>262144||!w.data)throw std::invalid_argument("Gemma TG global merge capacity/arena");
    const int count=(capacity+127)/128;
    if(count>MaxParts||std::size_t(Heads)*count*(Dim+2)*4>w.bytes)
        throw std::invalid_argument("Gemma TG global merge AoS extent");
    return count;
}
}
namespace strata::gemma4::native {
void global_t1_merge_only(float* output,const int* position,int capacity,AttentionWorkspace w,cudaStream_t stream){
    const int parts=detail_global_merge::parts(capacity,w);
    if(!output||!position||!stream||output==w.data)throw std::invalid_argument("Gemma TG global merge operands/stream");
    detail_global_merge::merge<<<dim3(32,4),128,0,stream>>>(static_cast<const float*>(w.data),output,position,capacity,parts);
    primitive::check_cuda(cudaGetLastError(),"Gemma existing-segment parallel global AoS merge only");
}
}
