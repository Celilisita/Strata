#pragma once
#include <cuda/atomic>

namespace strata::qwen35::native::detail {
__device__ inline float4 global_coherent4(const float *source){
    float4 value;
    // Bypass the per-SM L1: this consumer reads other CTAs' current partials.
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(value.x),"=f"(value.y),"=f"(value.z),"=f"(value.w)
                 : "l"(source) : "memory");
    return value;
}
// Every producer fences its own stores. The leader publishes a release ticket
// after the CTA barrier; the last leader acquires all preceding release tickets.
// No CTA spins or waits for an unscheduled producer.
__device__ inline void finish_split3(const float *partials,float *output,unsigned *counters,
                                    int rows,int first_row,int tid){
    __shared__ unsigned ticket;
    __threadfence();
    __syncthreads();
    if(tid==0){
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> counter(counters[first_row/64]);
        ticket=counter.fetch_add(1,cuda::memory_order_acq_rel);
    }
    __syncthreads();
    if(ticket!=2)return;
    const size_t plane=size_t(rows)*32;
    for(int i=tid;i<32*16;i+=256){
        const size_t index=size_t(i/16)*rows+first_row+(i%16)*4;
        const float4 a=global_coherent4(partials+index);
        const float4 b=global_coherent4(partials+plane+index);
        const float4 c=global_coherent4(partials+2*plane+index);
        const float4 value=make_float4(
            __fadd_rn(__fadd_rn(a.x,b.x),c.x),__fadd_rn(__fadd_rn(a.y,b.y),c.y),
            __fadd_rn(__fadd_rn(a.z,b.z),c.z),__fadd_rn(__fadd_rn(a.w,b.w),c.w));
        reinterpret_cast<float4 *>(output+index)[0]=value;
    }
    __syncthreads();
    if(tid==0){
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> counter(counters[first_row/64]);
        counter.store(0,cuda::memory_order_release);
    }
}
}
