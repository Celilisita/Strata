#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>

namespace strata::gemma4::native::detail_global_target_rows {
constexpr int Dim=512,Warps=4,Partition=128;
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float read_q8(const void* data,size_t at,int i){
    const auto* p=static_cast<const unsigned char*>(data)+at+(i/32)*34;
    return __half2float(*reinterpret_cast<const half*>(p))*float(reinterpret_cast<const signed char*>(p+2)[i%32]);
}
// Same four warp key stripes and row arithmetic as the original global segment.
// Raw K/V decoding is shared only across the two/three target query rows.
template<int Rows> __global__ void segment_rows(const float* query,float* partial,CacheView c,const int* position,int heads,int parts){
    static_assert(Rows==2||Rows==3,"Global target row extent");
    __shared__ float vectors[Rows][Warps*Dim],m[Rows][Warps],z[Rows][Warps];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.x,part=blockIdx.y;
    const int start=*position,begin=part*Partition,kh=head/(heads/c.heads);
    float q[Rows][Dim/32],a[Rows][Dim/32]{},top[Rows],denominator[Rows]{};
    int qpos[Rows],end=-1;bool valid[Rows];
    #pragma unroll
    for(int r=0;r<Rows;++r){
        const long long absolute=static_cast<long long>(start)+r;
        valid[r]=absolute>=0&&absolute<c.capacity;qpos[r]=valid[r]?int(absolute):-1;
        end=max(end,qpos[r]);top[r]=-CUDART_INF_F;
        #pragma unroll
        for(int j=0;j<Dim/32;++j)q[r][j]=valid[r]?query[(size_t(r)*heads+head)*Dim+lane+32*j]:0.f;
    }
    end=min(end,begin+Partition-1);int slot=(begin+warp)%c.capacity;
    for(int key=begin+warp;key<=end;key+=Warps){
        const size_t at=(size_t(slot)*c.heads+kh)*(Dim/32)*34;
        float score[Rows]{},old[Rows],current[Rows];
        #pragma unroll
        for(int j=0;j<Dim/32;++j){
            const float k=read_q8(c.key,at,lane+32*j);
            #pragma unroll
            for(int r=0;r<Rows;++r)if(valid[r]&&key<=qpos[r])score[r]=fmaf(q[r][j],k,score[r]);
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)if(valid[r]&&key<=qpos[r]){
            score[r]=sum(score[r]);const float next=fmaxf(top[r],score[r]);
            old[r]=expf(top[r]-next);current[r]=expf(score[r]-next);top[r]=next;
        }
        #pragma unroll
        for(int j=0;j<Dim/32;++j){
            const float v=read_q8(c.value,at,lane+32*j);
            #pragma unroll
            for(int r=0;r<Rows;++r)if(valid[r]&&key<=qpos[r])a[r][j]=fmaf(current[r],v,old[r]*a[r][j]);
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)if(valid[r]&&key<=qpos[r])denominator[r]=fmaf(denominator[r],old[r],current[r]);
        slot+=Warps;if(slot>=c.capacity)slot-=c.capacity;
    }
    #pragma unroll
    for(int r=0;r<Rows;++r){
        if(!lane){m[r][warp]=top[r];z[r][warp]=denominator[r];}
        #pragma unroll
        for(int j=0;j<Dim/32;++j)vectors[r][warp*Dim+lane+32*j]=a[r][j];
    }
    __syncthreads();
    #pragma unroll
    for(int r=0;r<Rows;++r){
        float* out=partial+((size_t(r)*heads+head)*parts+part)*(Dim+2);
        if(!valid[r]){for(int i=threadIdx.x;i<Dim+2;i+=128)out[i]=CUDART_NAN_F;}
        else{
            const float all=fmaxf(fmaxf(m[r][0],m[r][1]),fmaxf(m[r][2],m[r][3]));float weights[Warps],total=0;
            #pragma unroll
            for(int w=0;w<Warps;++w){weights[w]=z[r][w]==0?0:expf(m[r][w]-all);total=fmaf(z[r][w],weights[w],total);}
            if(!threadIdx.x){out[0]=all;out[1]=total;}
            for(int i=threadIdx.x;i<Dim;i+=128){float value=0;
                #pragma unroll
                for(int w=0;w<Warps;++w)value=fmaf(weights[w],vectors[r][w*Dim+i],value);
                out[2+i]=value;}
        }
    }
}
}
