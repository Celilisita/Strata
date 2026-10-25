#include "sm120_mtp_phase_timing.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace strata::gemma4::native::sampled_mtp {
namespace {
__global__ void stamp(std::uint64_t* ticks,int slot){
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    ticks[slot]=static_cast<std::uint64_t>(now);
}
}
void phase_stamp(std::uint64_t* ticks,int slot,cudaStream_t stream){
    if(!ticks||slot<0||slot>=5||!stream)throw std::invalid_argument("MTP phase stamp span/stream");
    stamp<<<1,1,0,stream>>>(ticks,slot);
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
}
