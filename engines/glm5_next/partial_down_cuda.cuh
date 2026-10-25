#pragma once
#include "partial_down_protocol.hpp"
#include "cold_gpu_protocol.hpp"
#include "hybrid_join.cuh"

namespace strata::glm5_next {
__device__ inline PartialDownPhase* partial_down_phase(const PartialDownBridge* bridge){
    return bridge?reinterpret_cast<PartialDownPhase*>(bridge->phase):nullptr;
}
__device__ inline bool partial_down_capability(const PartialDownBridge* bridge,uint64_t epoch){
    return bridge&&bridge->magic==partial_down_bridge_magic&&bridge->owner_epoch==epoch&&bridge->bank_generation&&
        bridge->phase&&!(bridge->phase%64)&&bridge->metadata&&!(bridge->metadata%64)&&bridge->layer==24;
}
// Only lookup thread0 calls this, before releasing the original packet sequence.
__device__ inline void partial_down_publish_candidate(const PartialDownBridge* bridge,ColdGpuLayer* meta,
    const int32_t* ids,const int32_t* slots,const int32_t* indices,uint32_t sequence,uint64_t epoch,int32_t* status){
    if(!partial_down_capability(bridge,epoch)||bridge->metadata!=reinterpret_cast<uintptr_t>(meta)||!sequence){atomicExch(status,1);return;}
    auto* phase=partial_down_phase(bridge);
    phase->decision_seq=phase->mid_ready_seq=phase->mid_consumed_seq=phase->error_seq=0;
    phase->decision=phase->partial_mask=phase->cpu_down_mask=phase->reader_error=phase->reserved=0;
    phase->down_keys[0]=phase->down_keys[1]=-1;
    phase->owner_epoch=epoch;phase->bank_generation=bridge->bank_generation;phase->ticket=sequence;
    bool exact=cold_gpu_layout_valid(*meta,epoch,24,2)&&meta->topology_fallback==0&&
        meta->first_row==bridge->first_row&&meta->original_missing_mask==partial_down_missing_mask&&
        meta->active_mask==partial_down_whole_mask&&meta->selected_count==2&&meta->selected_keys[0]==22&&meta->selected_keys[1]==70;
    for(unsigned i=0;exact&&i<10;++i)exact=ids[partial_down_gu_rank(i)]==partial_down_cpu_id(i)&&slots[partial_down_gu_rank(i)]<0;
    exact=exact&&ids[6]==70&&ids[10]==22&&slots[6]==int(meta->first_row+1)&&slots[10]==int(meta->first_row)&&
        cold_gpu_alias_valid(*meta,57)&&cold_gpu_alias_valid(*meta,234);
    bool index_valid=indices;
    for(unsigned i=0;index_valid&&i<4;++i)index_valid=indices[i]==int(meta->first_row+(i==1||i==2))&&indices[i+4]==int(i==1||i==2?234:57);
    if(!index_valid){atomicExch(status,1);return;}
    if(exact){phase->partial_mask=partial_down_role_mask;phase->cpu_down_mask=partial_down_cpu_mask;
        phase->down_keys[0]=57;phase->down_keys[1]=234;}
}
__device__ inline bool partial_down_ticket_valid(const PartialDownBridge* bridge,const uint32_t* state,
    const deepseek_v4::DeviceMissPacket* packet,uint64_t epoch){
    if(!partial_down_capability(bridge,epoch)||!state||!state[0]||!state[1])return false;
    const auto* phase=partial_down_phase(bridge);
    return phase->owner_epoch==epoch&&phase->bank_generation==bridge->bank_generation&&phase->ticket==state[1]&&
        packet->owner_epoch==epoch&&packet->layer==24&&packet->native_request==5&&packet->routed_count==16;
}
__device__ inline void partial_down_device_error(PartialDownPhase* phase,int32_t* status,uint32_t ticket){
    atomicExch(status,1);
    if(phase){HybridWord(phase->reader_error).store(ticket,cuda::memory_order_release);}
}
__device__ inline bool partial_down_wait(const PartialDownBridge* bridge,const uint32_t* state,
    deepseek_v4::DeviceMissPacket* packet,int32_t* status,uint64_t epoch,uint64_t timeout,bool mid){
    if(!state[0])return false;
    if(!partial_down_capability(bridge,epoch)){atomicExch(status,1);return false;}
    auto* phase=partial_down_phase(bridge);const uint32_t ticket=state[1];
    if(!partial_down_ticket_valid(bridge,state,packet,epoch)){partial_down_device_error(phase,status,ticket);return false;}
    const uint64_t begin=hybrid_clock_ns();
    while(!*status){
        if(HybridWord(phase->error_seq).load(cuda::memory_order_acquire)==ticket||
            (HybridWord(hybrid_plain(packet->response)).load(cuda::memory_order_acquire)==ticket&&packet->error)){
            partial_down_device_error(phase,status,ticket);return false;}
        if(HybridWord(phase->decision_seq).load(cuda::memory_order_acquire)==ticket){
            if(phase->decision==partial_down_stock)return false;
            if(phase->decision!=partial_down_selected||phase->partial_mask!=partial_down_role_mask||
                phase->cpu_down_mask!=partial_down_cpu_mask||phase->down_keys[0]!=57||phase->down_keys[1]!=234){
                partial_down_device_error(phase,status,ticket);return false;}
            if(!mid||HybridWord(phase->mid_ready_seq).load(cuda::memory_order_acquire)==ticket)return true;
        }
        if(hybrid_clock_ns()-begin>=timeout){partial_down_device_error(phase,status,ticket);return false;}
        __nanosleep(1000);
    }
    return false;
}
static __global__ void partial_down_copy_rows(const PartialDownBridge* bridge,const uint32_t* state,
    deepseek_v4::DeviceMissPacket* packet,int32_t* status,uint64_t epoch,uint64_t timeout){
    __shared__ bool selected;
    if(threadIdx.x==0){selected=partial_down_wait(bridge,state,packet,status,epoch,timeout,false);
        if(selected){const auto& meta=*reinterpret_cast<const ColdGpuLayer*>(bridge->metadata);
            selected=cold_gpu_layout_valid(meta,epoch,24,2)&&meta.first_row==bridge->first_row&&
                meta.active_mask==partial_down_whole_mask&&meta.selected_keys[0]==22&&meta.selected_keys[1]==70&&
                cold_gpu_alias_valid(meta,57)&&cold_gpu_alias_valid(meta,234);
            if(!selected)partial_down_device_error(partial_down_phase(bridge),status,state[1]);}}
    __syncthreads();if(!selected)return;
    const auto& meta=*reinterpret_cast<const ColdGpuLayer*>(bridge->metadata);
    const unsigned index=blockIdx.y;const int key=index?234:57;const size_t words=size_t(meta.stride[2]/sizeof(uint4));
    const auto* source=reinterpret_cast<const uint4*>(meta.alias[2][key]);
    auto* destination=reinterpret_cast<uint4*>(meta.bank_base[2]+uint64_t(meta.first_row+index)*meta.stride[2]);
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)destination[i]=source[i];
}
static __global__ void partial_down_read_mid(const PartialDownBridge* bridge,const uint32_t* state,
    deepseek_v4::DeviceMissPacket* packet,int32_t* status,uint64_t epoch,uint64_t timeout,float* output){
    __shared__ bool selected;
    if(threadIdx.x==0)selected=partial_down_wait(bridge,state,packet,status,epoch,timeout,true);
    __syncthreads();
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=8192)return;
    if(*status){output[i]=__int_as_float(0x7fc00000);return;}
    if(!selected){output[i]=0;return;}
    auto* phase=partial_down_phase(bridge);
    if(HybridWord(phase->mid_ready_seq).load(cuda::memory_order_acquire)!=state[1]){
        partial_down_device_error(phase,status,state[1]);output[i]=__int_as_float(0x7fc00000);return;}
    const unsigned row=i/2048,col=i%2048;
    output[i]=reinterpret_cast<const volatile float*>(packet->contribution)[size_t(partial_down_rank(row))*4096+col];
}
// Separate same-stream kernel: every mid read CTA has retired before this word.
static __global__ void partial_down_mid_consumed(const PartialDownBridge* bridge,const uint32_t* state,int32_t* status){
    if(threadIdx.x||blockIdx.x||*status||!state[0])return;auto* phase=partial_down_phase(bridge);
    if(phase->ticket==state[1]&&HybridWord(phase->decision_seq).load(cuda::memory_order_acquire)==state[1]&&
        phase->decision==partial_down_selected&&HybridWord(phase->mid_ready_seq).load(cuda::memory_order_acquire)==state[1]){
        __threadfence_system();HybridWord(phase->mid_consumed_seq).store(state[1],cuda::memory_order_release);}
}
__device__ inline bool partial_down_join_valid(const PartialDownBridge* bridge,const uint32_t* state,
    deepseek_v4::DeviceMissPacket* packet,int32_t* status,uint64_t epoch){
    if(bridge&&*status)return false;
    if(!bridge||!state||!state[0])return true;
    if(!partial_down_capability(bridge,epoch)){atomicExch(status,1);return false;}
    if(!partial_down_ticket_valid(bridge,state,packet,epoch)){partial_down_device_error(partial_down_phase(bridge),status,state[1]);return false;}
    const auto* phase=partial_down_phase(bridge);const uint32_t ticket=state[1];
    if(HybridWord(const_cast<uint32_t&>(phase->error_seq)).load(cuda::memory_order_acquire)==ticket||
        HybridWord(const_cast<uint32_t&>(phase->reader_error)).load(cuda::memory_order_acquire)==ticket||
        HybridWord(const_cast<uint32_t&>(phase->decision_seq)).load(cuda::memory_order_acquire)!=ticket){atomicExch(status,1);return false;}
    if(phase->decision==partial_down_stock)return true;
    if(phase->decision!=partial_down_selected||phase->partial_mask!=partial_down_role_mask||packet->cpu_mask!=partial_down_cpu_mask||
        HybridWord(const_cast<uint32_t&>(phase->mid_consumed_seq)).load(cuda::memory_order_acquire)!=ticket){atomicExch(status,1);return false;}
    return true;
}
__device__ inline bool partial_down_value(const PartialDownBridge* bridge,const uint32_t* state,unsigned rank,
    unsigned column,const float* partial,float& value){
    if(!bridge||!state||!state[0])return false;const auto* phase=partial_down_phase(bridge);
    if(phase->decision!=partial_down_selected)return false;
    for(unsigned i=0;i<4;++i)if(rank==partial_down_rank(i)){value=partial[size_t(i)*4096+column];return true;}
    return false;
}
}
