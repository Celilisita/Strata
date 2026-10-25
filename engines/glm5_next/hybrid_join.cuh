#pragma once
#include "../deepseek_v4/device_cache_protocol.hpp"
#include "critical_intervals.cuh"
#include "causal_phases.cuh"
#include <cuda/atomic>
namespace strata::glm5_next {
// Match D4 native publish/join: one device ticket, system release/acquire,
// one existing mapped packet. No mapped access on the all-hit branch.
using HybridWord = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
__device__ inline uint32_t& hybrid_plain(volatile uint32_t& word) {
    return const_cast<uint32_t&>(word);
}
__device__ inline uint64_t hybrid_clock_ns() {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}
template<bool SampleIntervals=false,bool CausalPhases=false>
__device__ inline bool hybrid_join(const uint32_t* state, int32_t* status,
    unsigned long long* counters, strata::deepseek_v4::DeviceMissPacket* packet,
    uint64_t epoch, int layer, uint64_t timeout) {
    if (!state || !state[0]) return true;
    const uint32_t ticket = state[1];
    if (threadIdx.x == 0 && !*status) {
        if (!ticket || packet->owner_epoch != epoch || int(packet->layer) != layer ||
            packet->native_request != 5) atomicExch(status,1);
        const uint64_t start = hybrid_clock_ns();
        bool waited=false;
        while (!*status && HybridWord(hybrid_plain(packet->response)).load(cuda::memory_order_acquire) != ticket) {
            if constexpr(SampleIntervals)waited=true;
            if (hybrid_clock_ns()-start >= timeout) {
                atomicExch(status,1);atomicAdd(counters+4,1ull);break;
            }
            __nanosleep(1000);
        }
        if (packet->error) atomicExch(status,1);
        uint64_t causal_ready=0;
        if constexpr(CausalPhases)if(!*status&&causal_phase_representative()&&causal_phase_sample_ticket(ticket))causal_ready=hybrid_clock_ns();
        if constexpr(SampleIntervals)if(!*status&&blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0)
            record_critical_join<true>(counters,waited?hybrid_clock_ns()-start:0);
        // Existing layer63 poll recording precedes the new aggregate atomics.
        if constexpr(CausalPhases)if(causal_ready)causal_phase_join_at(counters,ticket,layer,epoch,start,causal_ready);
    }
    __syncthreads();
    if (*status) return false;
    (void) HybridWord(hybrid_plain(packet->response)).load(cuda::memory_order_acquire);
    return true;
}
}
