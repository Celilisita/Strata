#pragma once
#include "router_prefetch_kernels.cuh"
namespace strata::deepseek_v4 {
// The input transform and eight router rows share one CTA. The final CTA
// selects both positions and publishes one immutable route descriptor.
__global__ void fused_router_prefetch(const float* raw,const float* norm,const float* hc_fn,
    const float* scale,const float* base,const __nv_bfloat16* weight,const float* bias,
    const int32_t* directory,float* scores,uint32_t* completed,
    RouterPrefetchPacket* packet,RouterPredictionRecord* records,const uint32_t* epochs,
    RouterPredictionParams p,int tokens,float* sentinel,uint64_t* clocks) {
    const int token=blockIdx.y,tid=threadIdx.x,warp=tid/32,lane=tid&31;
    if(clocks&&blockIdx.x==0&&token==0&&tid==0)clocks[0]=prediction_clock();
    __shared__ float reduction[8],hc[4],normalized[4096];
    __shared__ float local_score[48],selected_score[12];
    __shared__ int local_id[48],last;
    __shared__ RouterPrefetchRoutes entry;
    const float* x=raw+token*16384;
    float sum=0;
    for(int i=tid;i<16384;i+=256)sum+=x[i]*x[i];
    const float raw_rms=rsqrtf(prediction_block_sum(sum,reduction)/16384+p.norm_eps);
    for(int channel=0;channel<4;++channel){
        sum=0;for(int i=tid;i<16384;i+=256)sum+=x[i]*hc_fn[channel*16384+i];
        const float mix=prediction_block_sum(sum,reduction)*raw_rms;
        if(tid==0)hc[channel]=1.f/(1.f+expf(-(mix*scale[0]+base[channel])))+p.hc_eps;
        __syncthreads();
    }
    sum=0;
    for(int i=tid;i<4096;i+=256){
        float value=0;
        for(int channel=0;channel<4;++channel)value+=x[channel*4096+i]*hc[channel];
        normalized[i]=value;sum+=value*value;
    }
    const float inverse=rsqrtf(prediction_block_sum(sum,reduction)/4096+p.norm_eps);
    for(int i=tid;i<4096;i+=256)normalized[i]*=inverse*norm[i];
    __syncthreads();
    if(clocks&&blockIdx.x==0&&token==0&&tid==0)clocks[1]=prediction_clock();
    const int expert=blockIdx.x*8+warp;
    float value=0;
    for(int i=lane;i<4096;i+=32)value+=__bfloat162float(weight[expert*4096+i])*normalized[i];
    value=prediction_sum(value);
    if(lane==0){const float softplus=value>20.f?value:log1pf(expf(value));
        scores[token*256+expert]=sqrtf(softplus)+(bias?bias[expert]:0);}
    __syncthreads();__threadfence();
    if(tid==0)last=atomicAdd(completed,1)==unsigned(32*tokens-1);
    __syncthreads();if(!last)return;
    if(clocks&&tid==0)clocks[2]=prediction_clock();
    if(tid==0){entry={};entry.epoch=*epochs-tokens+1;entry.layer=p.layer;entry.tokens=tokens;
        entry.selected[0]=entry.selected[1]=-1;}
    __syncthreads();
    for(int position=0;position<tokens;++position){
        float score=scores[position*256+tid];
        for(int rank=0;rank<6;++rank){
            float best=score;int candidate=tid;
            for(int offset=16;offset;offset>>=1){
                const float other=__shfl_down_sync(0xffffffff,best,offset);
                const int other_id=__shfl_down_sync(0xffffffff,candidate,offset);
                if(lane+offset<32&&(other>best||(other==best&&other_id<candidate)))
                    {best=other;candidate=other_id;}
            }
            if(lane==0){local_score[warp*6+rank]=best;local_id[warp*6+rank]=candidate;}
            const int chosen=__shfl_sync(0xffffffff,candidate,0);
            if(tid==chosen)score=-INFINITY;
        }
        __syncthreads();
        if(warp==0){
            float a=lane<32?local_score[lane]:-INFINITY;
            float b=lane<16?local_score[lane+32]:-INFINITY;
            int aid=lane<32?local_id[lane]:256;
            int bid=lane<16?local_id[lane+32]:256;
            for(int rank=0;rank<6;++rank){
                float best=a;int candidate=aid;
                if(b>best||(b==best&&bid<candidate)){best=b;candidate=bid;}
                for(int offset=16;offset;offset>>=1){
                    const float other=__shfl_down_sync(0xffffffff,best,offset);
                    const int other_id=__shfl_down_sync(0xffffffff,candidate,offset);
                    if(lane+offset<32&&(other>best||(other==best&&other_id<candidate)))
                        {best=other;candidate=other_id;}
                }
                const int chosen=__shfl_sync(0xffffffff,candidate,0);
                if(lane==0){const bool missing=directory[chosen]<0;
                    entry.ids[position*6+rank]=missing?chosen:-1;
                    selected_score[position*6+rank]=missing?best:-INFINITY;
                    if(missing)entry.masks[position]|=1u<<rank;
                    if(records&&entry.epoch+position&&entry.epoch+position<=8192){
                        auto& record=records[(entry.epoch+position-1)*43+p.layer];
                        record.top[2][rank]=chosen;record.score[2][rank]=best;}
                }
                if(aid==chosen)a=-INFINITY;
                if(bid==chosen)b=-INFINITY;
            }
            if(lane==0&&records&&entry.epoch+position&&entry.epoch+position<=8192){
                auto& record=records[(entry.epoch+position-1)*43+p.layer];
                record.missing[2]=entry.masks[position];record.end[2]=prediction_clock();}
        }
        __syncthreads();
    }
    if(tid==0){for(int priority=0;priority<2;++priority){int rank=-1;
            for(int r=0;r<tokens*6;++r)
                if(entry.ids[r]>=0&&entry.ids[r]!=entry.selected[0]&&
                   (rank<0||selected_score[r]>selected_score[rank]))rank=r;
            if(rank>=0)entry.selected[priority]=entry.ids[rank];}
        if(entry.masks[0]|entry.masks[1])publish_prefetch(packet,entry);
        *sentinel=0;atomicExch(completed,0);
        if(clocks)clocks[3]=prediction_clock();}
}
}
