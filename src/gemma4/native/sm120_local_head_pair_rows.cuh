#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
namespace strata::gemma4::native::detail_local_head_pair_rows {
constexpr int Dim=256,Warps=4,Partition=128,HeadPair=2;
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float read_q8(const void* data,size_t at,int i){
    const auto* p=static_cast<const unsigned char*>(data)+at+(i/32)*34;
    return __half2float(*reinterpret_cast<const half*>(p))*float(reinterpret_cast<const signed char*>(p+2)[i%32]);
}
// One KV head owns its two Q heads. Physical warps are rotated separately per
// target row; original row partitions and four F32 recurrence streams remain.
template<int Rows> __global__ void segment_rows(const float* query,float* partial,CacheView c,const int* position,int heads,int parts){
    static_assert(Rows==2||Rows==3,"Local target row extent");
    __shared__ float vectors[Rows][HeadPair][Warps*Dim],m[Rows][HeadPair][Warps],z[Rows][HeadPair][Warps];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,kh=blockIdx.x,part=blockIdx.y;
    const int start=*position,commonfirst=start>1023?start-1023:0,begin=commonfirst+part*Partition;
    float q[Rows][HeadPair][Dim/32],a[Rows][HeadPair][Dim/32]{};
    float top[Rows][HeadPair],denominator[Rows][HeadPair]{};
    int rowbegin[Rows],rowend[Rows],logicalwarp[Rows],end=-1;bool valid[Rows];
    #pragma unroll
    for(int r=0;r<Rows;++r){
        const long long absolute=static_cast<long long>(start)+r;
        valid[r]=absolute>=0&&absolute<=2147483647LL;
        const int qpos=valid[r]?int(absolute):-1,first=valid[r]?max(0,qpos-1023):commonfirst;
        rowbegin[r]=first+part*Partition;rowend[r]=valid[r]?min(qpos,rowbegin[r]+Partition-1):-1;
        logicalwarp[r]=(warp-(first-commonfirst)+Warps)%Warps;end=max(end,rowend[r]);
        #pragma unroll
        for(int h=0;h<HeadPair;++h){
            top[r][h]=-CUDART_INF_F;
            #pragma unroll
            for(int j=0;j<Dim/32;++j)
                q[r][h][j]=valid[r]?query[(size_t(r)*heads+HeadPair*kh+h)*Dim+lane+32*j]:0.f;
        }
    }
    int slot=(begin+warp)%c.capacity;
    for(int key=begin+warp;key<=end;key+=Warps){
        const size_t at=(size_t(slot)*c.heads+kh)*(Dim/32)*34;
        float score[Rows][HeadPair]{},old[Rows][HeadPair],current[Rows][HeadPair];
        #pragma unroll
        for(int j=0;j<Dim/32;++j){
            const float k=read_q8(c.key,at,lane+32*j);
            #pragma unroll
            for(int r=0;r<Rows;++r)if(valid[r]&&key>=rowbegin[r]&&key<=rowend[r]){
                #pragma unroll
                for(int h=0;h<HeadPair;++h)score[r][h]=fmaf(q[r][h][j],k,score[r][h]);
            }
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)if(valid[r]&&key>=rowbegin[r]&&key<=rowend[r]){
            #pragma unroll
            for(int h=0;h<HeadPair;++h){
                score[r][h]=sum(score[r][h]);const float next=fmaxf(top[r][h],score[r][h]);
                old[r][h]=expf(top[r][h]-next);current[r][h]=expf(score[r][h]-next);top[r][h]=next;
            }
        }
        #pragma unroll
        for(int j=0;j<Dim/32;++j){
            const float v=read_q8(c.value,at,lane+32*j);
            #pragma unroll
            for(int r=0;r<Rows;++r)if(valid[r]&&key>=rowbegin[r]&&key<=rowend[r]){
                #pragma unroll
                for(int h=0;h<HeadPair;++h)a[r][h][j]=fmaf(current[r][h],v,old[r][h]*a[r][h][j]);
            }
        }
        #pragma unroll
        for(int r=0;r<Rows;++r)if(valid[r]&&key>=rowbegin[r]&&key<=rowend[r]){
            #pragma unroll
            for(int h=0;h<HeadPair;++h)denominator[r][h]=fmaf(denominator[r][h],old[r][h],current[r][h]);
        }
        slot+=Warps;if(slot>=c.capacity)slot-=c.capacity;
    }
    #pragma unroll
    for(int r=0;r<Rows;++r){
        const int w=logicalwarp[r];
        #pragma unroll
        for(int h=0;h<HeadPair;++h){
            if(!lane){m[r][h][w]=top[r][h];z[r][h][w]=denominator[r][h];}
            #pragma unroll
            for(int j=0;j<Dim/32;++j)vectors[r][h][w*Dim+lane+32*j]=a[r][h][j];
        }
    }
    __syncthreads();
    #pragma unroll
    for(int r=0;r<Rows;++r){
        #pragma unroll
        for(int h=0;h<HeadPair;++h){
            float* out=partial+((size_t(r)*heads+HeadPair*kh+h)*parts+part)*(Dim+2);
            if(!valid[r]){for(int i=threadIdx.x;i<Dim+2;i+=128)out[i]=CUDART_NAN_F;}
            else{
                const float all=fmaxf(fmaxf(m[r][h][0],m[r][h][1]),fmaxf(m[r][h][2],m[r][h][3]));float weights[Warps],total=0;
                #pragma unroll
                for(int w=0;w<Warps;++w){weights[w]=z[r][h][w]==0?0:expf(m[r][h][w]-all);total=fmaf(z[r][h][w],weights[w],total);}
                if(!threadIdx.x){out[0]=all;out[1]=total;}
                for(int i=threadIdx.x;i<Dim;i+=128){float value=0;
                    #pragma unroll
                    for(int w=0;w<Warps;++w)value=fmaf(weights[w],vectors[r][h][w*Dim+i],value);
                    out[2+i]=value;}
            }
        }
    }
}
}
