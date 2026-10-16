#pragma once
#include "router_prediction.hpp"
#include <cuda_bf16.h>
namespace strata::deepseek_v4 {
__device__ inline uint64_t prediction_clock() {unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;}
__device__ inline float prediction_sum(float value) {
    for(int offset=16;offset;offset>>=1)value+=__shfl_down_sync(0xffffffff,value,offset);return value;
}
__device__ inline float prediction_block_sum(float value,float* shared) {
    __syncthreads();
    value=prediction_sum(value);if((threadIdx.x&31)==0)shared[threadIdx.x/32]=value;__syncthreads();
    value=threadIdx.x<8?shared[threadIdx.x]:0;value=prediction_sum(value);
    if(threadIdx.x==0)shared[0]=value;__syncthreads();
    const float result=shared[0];__syncthreads();return result;
}
__global__ void prediction_normalize(const float* raw,const float* merged,const float* norm,
    const float* hc_fn,const float* scale,const float* base,float* normalized,
    RouterPredictionRecord* records,const uint32_t* epochs,RouterPredictionParams p,int tokens,uint64_t* clocks) {
    const int token=blockIdx.x;
    if(clocks&&p.oracle==2&&token==0&&threadIdx.x==0)clocks[0]=prediction_clock();const uint32_t epoch=*epochs-tokens+1+token;
    if(records&&threadIdx.x==0&&epoch&&epoch<=8192) {
        auto& record=records[(epoch-1)*43+p.layer];for(int v=0;v<3;++v)record.start[v]=prediction_clock();
    }
    __shared__ float reduction[8],hc[4];
    const float* x=raw+token*16384;
    float sum=0;for(int i=threadIdx.x;i<16384;i+=256)sum+=x[i]*x[i];
    const float raw_rms=rsqrtf(prediction_block_sum(sum,reduction)/16384+p.norm_eps);
    for(int channel=0;channel<4;++channel) {
        sum=0;for(int i=threadIdx.x;i<16384;i+=256)sum+=x[i]*hc_fn[channel*16384+i];
        const float mix=prediction_block_sum(sum,reduction)*raw_rms;
        if(threadIdx.x==0)hc[channel]=1.f/(1.f+expf(-(mix*scale[0]+base[channel])))+p.hc_eps;
        __syncthreads();
    }
    for(int variant=0;variant<3;++variant) {
        if(p.oracle==2&&variant!=2)continue;
        float* out=normalized+(variant*2+token)*4096;sum=0;
        for(int i=threadIdx.x;i<4096;i+=256) {
            float value=0;
            if(variant==1)value=merged[token*4096+i];
            else for(int channel=0;channel<4;++channel)value+=x[channel*4096+i]*(variant==0?.25f:hc[channel]);
            out[i]=value;sum+=value*value;
        }
        const float inverse=rsqrtf(prediction_block_sum(sum,reduction)/4096+p.norm_eps);
        for(int i=threadIdx.x;i<4096;i+=256)out[i]*=inverse*norm[i];
        __syncthreads();
    }
    if(clocks&&p.oracle==2&&token==0&&threadIdx.x==0)clocks[1]=prediction_clock();
}
__global__ void prediction_project(const __nv_bfloat16* weight,const float* x,const float* bias,
    float* scores,RouterPredictionRecord* records,const uint32_t* epochs,RouterPredictionParams p,int tokens) {
    const int variant=p.oracle==1?3:(p.oracle==2?2:int(blockIdx.y)/tokens),token=int(blockIdx.y)%tokens;
    const int expert=int(blockIdx.x)*4+int(threadIdx.x)/32,lane=threadIdx.x&31;
    const float* activation=p.oracle==1?x+token*4096:x+(variant*2+token)*4096;
    if(p.oracle==1&&records&&blockIdx.x==0&&threadIdx.x==0) {
        const uint32_t epoch=*epochs-tokens+1+token;if(epoch&&epoch<=8192)records[(epoch-1)*43+p.layer].start[3]=prediction_clock();
    }
    float value=0;for(int i=lane;i<4096;i+=32)value+=__bfloat162float(weight[expert*4096+i])*activation[i];
    value=prediction_sum(value);
    if(lane==0) {
        const float softplus=value>20.f?value:log1pf(expf(value));
        scores[(variant*2+token)*256+expert]=sqrtf(softplus)+(bias?bias[expert]:0);
    }
}
__global__ void prediction_select(const float* scores,const int32_t* directory,RouterPredictionRecord* records,
    const uint32_t* epochs,RouterPredictionParams p,int tokens,float* sentinel) {
    const int variant=p.oracle?3:int(blockIdx.x)/tokens,token=int(blockIdx.x)%tokens;
    const uint32_t epoch=*epochs-tokens+1+token;if(!epoch||epoch>8192)return;
    auto& record=records[(epoch-1)*43+p.layer];const int id=threadIdx.x;
    float value=scores[(variant*2+token)*256+id];
    __shared__ float top_value[8];__shared__ int top_id[8],winner;__shared__ uint32_t missing;
    if(id==0)missing=0;__syncthreads();
    for(int rank=0;rank<10;++rank) {
        float best=value;int selected=id;
        for(int offset=16;offset;offset>>=1) {
            const float other=__shfl_down_sync(0xffffffff,best,offset);const int candidate=__shfl_down_sync(0xffffffff,selected,offset);
            if((id&31)+offset<32&&(other>best||(other==best&&candidate<selected))) {best=other;selected=candidate;}
        }
        if((id&31)==0){top_value[id/32]=best;top_id[id/32]=selected;}__syncthreads();
        if(id==0) {
            int lane=0;for(int i=1;i<8;++i)if(top_value[i]>top_value[lane]||(top_value[i]==top_value[lane]&&top_id[i]<top_id[lane]))lane=i;
            winner=top_id[lane];record.top[variant][rank]=winner;record.score[variant][rank]=top_value[lane];
            if(directory[winner]<0)missing|=1u<<rank;
        }
        __syncthreads();if(id==winner)value=-INFINITY;__syncthreads();
    }
    if(id==0){record.missing[variant]=missing;record.end[variant]=prediction_clock();*sentinel=0;}
}
}
