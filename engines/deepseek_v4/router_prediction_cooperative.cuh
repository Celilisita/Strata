#pragma once
#include "router_prefetch_kernels.cuh"
#include <cooperative_groups.h>
namespace strata::deepseek_v4 {
namespace cg=cooperative_groups;
// One cooperative launch: distributed mHC reductions, merged RMS, router
// GEMV, two-position top-six and mapped publication. No intermediate launch.
__global__ void cooperative_router_prefetch(const float* raw,const float* norm,const float* hc_fn,
    const float* scale,const float* base,const __nv_bfloat16* weight,const float* bias,
    const int32_t* directory,float* merged,float* partial,float* norm_partial,
    float* scores,uint32_t* completed,RouterPrefetchPacket* packet,
    RouterPredictionRecord* records,const uint32_t* epochs,RouterPredictionParams p,
    int tokens,float* sentinel,uint64_t* clocks) {
    const int token=blockIdx.y,block=blockIdx.x,tid=threadIdx.x,warp=tid/32,lane=tid&31;
    const auto grid=cg::this_grid();
    if(clocks&&block==0&&token==0&&tid==0)clocks[0]=prediction_clock();
    __shared__ float reduction[8],hc[4],inverse;
    __shared__ float local_score[48],selected_score[12];
    __shared__ int local_id[48],last;
    __shared__ RouterPrefetchRoutes entry;
    const float* x=raw+token*16384;
    for(int component=0;component<5;++component){
        float sum=0;
        for(int i=block*512+tid;i<(block+1)*512;i+=256)
            sum+=component==0?x[i]*x[i]:x[i]*hc_fn[(component-1)*16384+i];
        const float value=prediction_block_sum(sum,reduction);
        if(tid==0)partial[(token*32+block)*5+component]=value;
    }
    grid.sync();
    if(tid==0){float totals[5]{};
        for(int i=0;i<32;++i)for(int component=0;component<5;++component)
            totals[component]+=partial[(token*32+i)*5+component];
        const float raw_rms=rsqrtf(totals[0]/16384+p.norm_eps);
        for(int channel=0;channel<4;++channel)
            hc[channel]=1.f/(1.f+expf(-(totals[channel+1]*raw_rms*scale[0]+base[channel])))+p.hc_eps;
    }
    __syncthreads();
    float square=0;
    if(tid<128){const int i=block*128+tid;float value=0;
        for(int channel=0;channel<4;++channel)value+=x[channel*4096+i]*hc[channel];
        merged[token*4096+i]=value;square=value*value;}
    const float sum=prediction_block_sum(square,reduction);
    if(tid==0)norm_partial[token*32+block]=sum;
    grid.sync();
    if(tid==0){float total=0;for(int i=0;i<32;++i)total+=norm_partial[token*32+i];
        inverse=rsqrtf(total/4096+p.norm_eps);}
    __syncthreads();
    if(tid<128){const int i=block*128+tid;merged[token*4096+i]*=inverse*norm[i];}
    grid.sync();
    if(clocks&&block==0&&token==0&&tid==0)clocks[1]=prediction_clock();
    const int expert=block*8+warp;
    float value=0;
    for(int i=lane;i<4096;i+=32)
        value+=__bfloat162float(weight[expert*4096+i])*merged[token*4096+i];
    value=prediction_sum(value);
    if(lane==0){const float softplus=value>20.f?value:log1pf(expf(value));
        scores[token*256+expert]=sqrtf(softplus)+(bias?bias[expert]:0);}
    __syncthreads();
    if(tid==0){cuda::atomic_ref<uint32_t,cuda::thread_scope_device> done(*completed);
        last=done.fetch_add(1,cuda::memory_order_acq_rel)==unsigned(32*tokens-1);}
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
            float a=local_score[lane],b=lane<16?local_score[lane+32]:-INFINITY;
            int aid=local_id[lane],bid=lane<16?local_id[lane+32]:256;
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
                if(aid==chosen)a=-INFINITY;if(bid==chosen)b=-INFINITY;
            }
            if(lane==0&&records&&entry.epoch+position&&entry.epoch+position<=8192){
                auto& record=records[(entry.epoch+position-1)*43+p.layer];
                record.missing[2]=entry.masks[position];record.end[2]=prediction_clock();}
        }
        __syncthreads();
    }
    if(tid==0){for(int priority=0;priority<2;++priority){int rank=-1;
            for(int r=0;r<tokens*6;++r)if(entry.ids[r]>=0&&entry.ids[r]!=entry.selected[0]&&
                (rank<0||selected_score[r]>selected_score[rank]))rank=r;
            if(rank>=0)entry.selected[priority]=entry.ids[rank];}
        if(entry.masks[0]|entry.masks[1])publish_prefetch(packet,entry);
        *sentinel=0;atomicExch(completed,0);if(clocks)clocks[3]=prediction_clock();}
}
}
