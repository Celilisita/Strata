#pragma once
#include "causal_phases.hpp"
namespace strata::glm5_next {
__device__ inline uint64_t causal_phase_clock_ns() {
    unsigned long long now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));return now;
}
__device__ inline bool causal_phase_representative() {
    return blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0;
}
__device__ inline void causal_phase_triplet(unsigned long long* counters,size_t at,uint64_t ns) {
    atomicAdd(counters+at,static_cast<unsigned long long>(ns));
    atomicMax(counters+at+1,static_cast<unsigned long long>(ns));atomicAdd(counters+at+2,1ull);
}
__device__ inline void causal_phase_publish_at(unsigned long long* counters,uint32_t ticket,int layer,uint64_t epoch,uint64_t ns) {
    if(!causal_phase_representative()||!causal_phase_sample_ticket(ticket))return;
    if(!causal_phase_layer_valid(layer)||!epoch||!ns){atomicAdd(counters+causal_phase_mismatches,1ull);return;}
    counters[causal_phase_scratch_publish]=ns;counters[causal_phase_scratch_tag]=causal_phase_tag(ticket,layer);
    counters[causal_phase_scratch_epoch]=epoch;atomicAdd(counters+causal_phase_publish_primary+causal_phase_group(layer),1ull);
}
template<bool Enabled>
__device__ inline void causal_phase_publish(unsigned long long* counters,uint32_t ticket,int layer,uint64_t epoch) {
    if constexpr(Enabled)if(causal_phase_representative()&&causal_phase_sample_ticket(ticket))
        causal_phase_publish_at(counters,ticket,layer,epoch,causal_phase_clock_ns());
}
__device__ inline void causal_phase_join_at(unsigned long long* counters,uint32_t ticket,int layer,uint64_t epoch,uint64_t entry,uint64_t ready) {
    if(!causal_phase_representative()||!causal_phase_sample_ticket(ticket))return;
    const uint64_t publish=counters[causal_phase_scratch_publish];
    if(!causal_phase_layer_valid(layer)||!epoch||counters[causal_phase_scratch_epoch]!=epoch||
        counters[causal_phase_scratch_tag]!=causal_phase_tag(ticket,layer)||!publish||entry<publish||ready<entry) {
        atomicAdd(counters+causal_phase_mismatches,1ull);return;
    }
    const size_t base=causal_phase_group(layer)?causal_phase_nextn_stats:causal_phase_primary_stats;
    causal_phase_triplet(counters,base,entry-publish);causal_phase_triplet(counters,base+3,ready-entry);
    causal_phase_triplet(counters,base+6,ready-publish);
}
}
