#pragma once
#include "router_prefetch.hpp"
#include "router_prediction_kernels.cuh"
#include <cuda/atomic>
namespace strata::deepseek_v4 {
__device__ void publish_prefetch(RouterPrefetchPacket* packet,const RouterPrefetchRoutes& entry) {
    cuda::atomic_ref<uint32_t,cuda::thread_scope_system> sequence(*const_cast<uint32_t*>(&packet->sequence));
    const uint32_t next=sequence.load(cuda::memory_order_relaxed)+1;
    packet->learned[next&1]=entry;sequence.store(next,cuda::memory_order_release);
}
// One block selects both positions, then publishes only missing IDs.
__global__ void prefetch_select(const float* scores,const int32_t* directory,
    RouterPrefetchPacket* packet,const uint32_t* epochs,int layer,int tokens,float* sentinel,uint64_t* clocks) {
    const int id=threadIdx.x;if(clocks&&id==0)clocks[2]=prediction_clock();__shared__ float warp_score[8],selected_score[12];
    __shared__ int warp_id[8],winner;__shared__ RouterPrefetchRoutes entry;
    if(id==0){entry={};entry.epoch=*epochs-tokens+1;entry.layer=layer;entry.tokens=tokens;entry.selected[0]=entry.selected[1]=-1;}
    __syncthreads();
    for(int token=0;token<tokens;++token){float value=scores[(4+token)*256+id];
        for(int rank=0;rank<6;++rank){float best=value;int selected=id;
            for(int offset=16;offset;offset>>=1){const float other=__shfl_down_sync(0xffffffff,best,offset);const int candidate=__shfl_down_sync(0xffffffff,selected,offset);
                if((id&31)+offset<32&&(other>best||(other==best&&candidate<selected))){best=other;selected=candidate;}}
            if((id&31)==0){warp_score[id/32]=best;warp_id[id/32]=selected;}__syncthreads();
            if(id==0){int w=0;for(int i=1;i<8;++i)if(warp_score[i]>warp_score[w]||(warp_score[i]==warp_score[w]&&warp_id[i]<warp_id[w]))w=i;
                winner=warp_id[w];const bool missing=directory[winner]<0;entry.ids[token*6+rank]=missing?winner:-1;
                if(missing)entry.masks[token]|=1u<<rank;selected_score[token*6+rank]=missing?warp_score[w]:-INFINITY;}
            __syncthreads();if(id==winner)value=-INFINITY;__syncthreads();
        }
    }
    if(id==0){for(int priority=0;priority<2;++priority){int rank=-1;for(int r=0;r<tokens*6;++r)
            if(entry.ids[r]>=0&&entry.ids[r]!=entry.selected[0]&&(rank<0||selected_score[r]>selected_score[rank]))rank=r;
            if(rank>=0)entry.selected[priority]=entry.ids[rank];}
        if(entry.masks[0]|entry.masks[1])publish_prefetch(packet,entry);*sentinel=0;if(clocks)clocks[3]=prediction_clock();}
}
__global__ void prefetch_hash(const int32_t* pending,const int32_t* draft,
    const int32_t* table0,const int32_t* table1,const int32_t* table2,
    const int32_t* directory0,const int32_t* directory1,const int32_t* directory2,
    RouterPrefetchPacket* packet,const uint32_t* epochs,bool both,float* sentinel) {
    if(threadIdx.x)return;const int token0=*pending,token1=both?*draft:0;
    if(token0<0||token0>=129280||(both&&(token1<0||token1>=129280)))return;
    cuda::atomic_ref<uint32_t,cuda::thread_scope_system> sequence(*const_cast<uint32_t*>(&packet->hash_sequence));
    const uint32_t next=sequence.load(cuda::memory_order_relaxed)+1;
    const int32_t* tables[]{table0,table1,table2};const int32_t* directories[]{directory0,directory1,directory2};
    for(int layer=0;layer<3;++layer){RouterPrefetchRoutes entry{};entry.epoch=*epochs+1;entry.layer=layer;entry.tokens=both?2:1;entry.source=both?2:1;
        for(int token=0;token<int(entry.tokens);++token)for(int rank=0;rank<6;++rank){const int expert=tables[layer][(token?token1:token0)*6+rank];
            const bool missing=expert>=0&&expert<256&&directories[layer][expert]<0;entry.ids[token*6+rank]=missing?expert:-1;if(missing)entry.masks[token]|=1u<<rank;}
        entry.selected[0]=entry.selected[1]=-1;packet->hash[next&1][layer]=entry;}
    sequence.store(next,cuda::memory_order_release);*sentinel=0;
}
}
