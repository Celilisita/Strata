#pragma once
#include "cold_gpu_protocol.hpp"
#include "cold_gpu_topology.hpp"
#include "cold_gpu_repeat_priority.hpp"
#include "cold_gpu_conditional4.hpp"
#include "../deepseek_v4/device_cache_protocol.hpp"
namespace strata::glm5_next {
// Mode3 keeps the original ticket/cache/copy protocol and only reselects K2.
__device__ inline void cold_gpu_select_repeat_priority(ColdGpuLayer* meta,const int32_t* ids,int32_t* slots,int32_t* status,uint64_t epoch,int layer){
    meta->active_mask=0;meta->original_missing_mask=0;meta->selected_count=0;
    for(int i=0;i<4;++i)meta->selected_keys[i]=-1;
    ++meta->lookup_calls;
    for(int rank=0;rank<16;++rank)if(slots[rank]<0)meta->original_missing_mask|=1u<<rank;
    if(*status||!cold_gpu_layout_valid(*meta,epoch,layer,2)){++meta->metadata_errors;++meta->ineligible_lookups;return;}
    if(!meta->original_missing_mask){++meta->allhit_lookups;return;}
    const auto result=cold_gpu_repeat_priority_selection(*meta,ids,meta->original_missing_mask);
    const auto& selected=result.selected;
    if(!selected.mask){++meta->ineligible_lookups;return;}
    for(uint32_t i=0;i<selected.count;++i)meta->selected_keys[i]=selected.keys[i];
    for(int rank=0;rank<16;++rank)if(selected.mask&(1u<<rank))
        for(uint32_t i=0;i<selected.count;++i)if(ids[rank]==selected.keys[i]){slots[rank]=int32_t(meta->first_row+i);meta->active_mask|=1u<<rank;}
    meta->selected_count=selected.count;++meta->eligible_lookups;
    meta->offloaded_occurrences+=cold_gpu_occurrences(meta->active_mask);
}
// Lookup thread0 only, after the original cache probe/capture/frequency work.
// The fixed reservation is phase-owned; per-ticket selection is2/4/0.
template<unsigned ReservedRows=2>
__device__ inline void cold_gpu_select(ColdGpuLayer* meta,const int32_t* ids,int32_t* slots,int32_t* status,uint64_t epoch,int layer){
    static_assert(ReservedRows==2||ReservedRows==4,"cold GPU reservation must be2or4");
    if constexpr(ReservedRows==2)if(meta->topology_fallback==3){
        cold_gpu_select_repeat_priority(meta,ids,slots,status,epoch,layer);return;
    }
    meta->active_mask=0;meta->original_missing_mask=0;meta->selected_count=0;
    for(int i=0;i<4;++i)meta->selected_keys[i]=-1;
    ++meta->lookup_calls;
    for(int rank=0;rank<16;++rank)if(slots[rank]<0)meta->original_missing_mask|=1u<<rank;
    if(*status||!cold_gpu_layout_valid(*meta,epoch,layer,ReservedRows)){++meta->metadata_errors;++meta->ineligible_lookups;return;}
    if(!meta->original_missing_mask){++meta->allhit_lookups;return;}
    int selected[4]{-1,-1,-1,-1},ranks[4]{-1,-1,-1,-1};
    // Keep the original lowest valid globally-unique key per position.
    for(int position=0;position<2;++position){int lowest=288;
        for(int rank=position*8;rank<position*8+8;++rank){const int key=ids[rank];if(slots[rank]>=0||!cold_gpu_alias_valid(*meta,key))continue;
            int occurrences=0;for(int other=0;other<16;++other)occurrences+=ids[other]==key;
            if(occurrences==1&&key<lowest){lowest=key;selected[position]=key;ranks[position]=rank;}}}
    unsigned count=2;bool topology=false;
    if(selected[0]<0||selected[1]<0){
        if constexpr(ReservedRows==2)if(meta->topology_fallback){
            const auto next=cold_gpu_topology_selection(*meta,ids,meta->original_missing_mask);
            count=next.count;topology=next.mask!=0;
            for(unsigned i=0;i<count;++i){selected[i]=next.keys[i];ranks[i]=next.ranks[i];}
        }
        if(!topology){++meta->ineligible_lookups;return;}
    }
    if constexpr(ReservedRows==4)if(cold_gpu_allow_four(*meta,ids,meta->original_missing_mask)){
        for(int position=0;position<2;++position){int lowest=288;
            for(int rank=position*8;rank<position*8+8;++rank){const int key=ids[rank];
                if(slots[rank]>=0||key==selected[position]||!cold_gpu_alias_valid(*meta,key))continue;
                int occurrences=0;for(int other=0;other<16;++other)occurrences+=ids[other]==key;
                if(occurrences==1&&key<lowest){lowest=key;selected[position+2]=key;ranks[position+2]=rank;}}}
        if(selected[2]>=0&&selected[3]>=0)count=4;
    }
    for(unsigned i=1;i<count;++i)for(unsigned j=i;j&&selected[j]<selected[j-1];--j){
        const int key=selected[j],rank=ranks[j];selected[j]=selected[j-1];ranks[j]=ranks[j-1];selected[j-1]=key;ranks[j-1]=rank;}
    if(!topology){
        for(unsigned i=0;i<count;++i){meta->selected_keys[i]=selected[i];slots[ranks[i]]=int32_t(meta->first_row+i);meta->active_mask|=1u<<ranks[i];}
    }else{
        for(unsigned i=0;i<count;++i)meta->selected_keys[i]=selected[i];
        for(int rank=0;rank<16;++rank)if(meta->original_missing_mask&(1u<<rank))
            for(unsigned i=0;i<count;++i)if(ids[rank]==selected[i]){slots[rank]=int32_t(meta->first_row+i);meta->active_mask|=1u<<rank;}
    }
    meta->selected_count=count;++meta->eligible_lookups;
    meta->offloaded_occurrences+=topology?cold_gpu_occurrences(meta->active_mask):count;
}
// One copy dispatch follows every attached lookup, including mask0. Both
// variants retain48 x(3*ReservedRows)/256 geometry and coalesced uint4 copies.
template<unsigned ReservedRows>
__device__ inline void cold_gpu_copy_body(ColdGpuLayer* meta,int32_t* status,uint64_t epoch,int layer){
    const uint32_t active=meta->active_mask,count=meta->selected_count;
    if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)++meta->copy_dispatches;
    if(!active)return;
    __shared__ bool valid;
    if(threadIdx.x==0){
        const bool topology=meta->topology_fallback!=0;
        const unsigned occurrences=topology?__popc(active):count;
        const bool shape=topology?(ReservedRows==2&&count>=1&&count<=2&&occurrences>=count&&occurrences<=2*count):
            ((count==2||count==4)&&__popc(active&255u)==int(count/2)&&__popc(active>>8)==int(count/2));
        valid=cold_gpu_layout_valid(*meta,epoch,layer,ReservedRows)&&shape&&count<=ReservedRows&&
            !(active>>16)&&!(active&~meta->original_missing_mask);
        if(valid)for(uint32_t i=0;i<count;++i)valid&=cold_gpu_alias_valid(*meta,meta->selected_keys[i])&&(!i||meta->selected_keys[i-1]<meta->selected_keys[i]);
    }
    __syncthreads();
    if(!valid){
        if(threadIdx.x==0)atomicExch(status,1);
        if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)++meta->metadata_errors;return;}
    if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)meta->copied_bytes+=uint64_t(count)*(meta->stride[0]+meta->stride[1]+meta->stride[2]);
    const unsigned role=blockIdx.y/ReservedRows,index=blockIdx.y%ReservedRows;
    if(index>=count)return; // A reserved4 ticket may safely select only2.
    const int key=meta->selected_keys[index];
    const auto* source=reinterpret_cast<const uint4*>(meta->alias[role][key]);
    auto* destination=reinterpret_cast<uint4*>(meta->bank_base[role]+uint64_t(meta->first_row+index)*meta->stride[role]);
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<meta->stride[role]/16;i+=uint64_t(gridDim.x)*blockDim.x)destination[i]=source[i];
}
__global__ void cold_gpu_copy_rows(ColdGpuLayer* meta,int32_t* status,uint64_t epoch,int layer){cold_gpu_copy_body<2>(meta,status,epoch,layer);}
__global__ void cold_gpu_copy_rows4(ColdGpuLayer* meta,int32_t* status,uint64_t epoch,int layer){cold_gpu_copy_body<4>(meta,status,epoch,layer);}
}
