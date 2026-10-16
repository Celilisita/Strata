#pragma once
#include "small_prefill_fused_reduce.cuh"

namespace strata::qwen35::native::detail {
// Last producer reduces in fixed plane order. No producer waits or spins.
__device__ inline void finish_q4_split5(const float *parts,float *output,unsigned *counters,int rows,int first,int tid){
    __shared__ unsigned ticket;
    __threadfence();__syncthreads();
    if(!tid){cuda::atomic_ref<unsigned,cuda::thread_scope_device> counter(counters[first/64]);ticket=counter.fetch_add(1,cuda::memory_order_acq_rel);}
    __syncthreads();if(ticket!=4)return;
    const size_t plane=size_t(rows)*32;
    for(int i=tid;i<32*16;i+=256){
        const int row=first+(i%16)*4;if(row>=rows)continue;
        const size_t at=size_t(i/16)*rows+row;float4 value=global_coherent4(parts+at);
        #pragma unroll
        for(int slice=1;slice<5;++slice){const float4 v=global_coherent4(parts+slice*plane+at);
            value=make_float4(__fadd_rn(value.x,v.x),__fadd_rn(value.y,v.y),__fadd_rn(value.z,v.z),__fadd_rn(value.w,v.w));}
        *reinterpret_cast<float4 *>(output+at)=value;
    }
    __syncthreads();
    if(!tid){cuda::atomic_ref<unsigned,cuda::thread_scope_device> counter(counters[first/64]);counter.store(0,cuda::memory_order_release);}
}
}
