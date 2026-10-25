#pragma once
#include "strata/kernels/radix_sampler.hpp"
#include <cuda_runtime.h>
#include <cstdint>
namespace strata::qwen35::native {
struct MtpProbCursor {uint64_t counter=0,epoch=0;};
struct MtpProbRow {
    uint64_t counter,epoch,seed;
    int vocabulary,kept,id;unsigned bad;
    int ids[64];double mass[64];
};
static_assert(sizeof(MtpProbCursor)==16&&sizeof(MtpProbRow)==808,"probability row ABI");
inline bool mtp_prob_params(const strata::kernels::SamplerParams &p){
    return !p.greedy&&p.temperature>0&&p.top_k>=1&&p.top_k<=64;
}
void mtp_prob_draft(const strata::kernels::RadixSampler&,int,const strata::kernels::SamplerParams&,
    const MtpProbCursor*,unsigned,MtpProbRow*,int*,unsigned*,cudaStream_t,int draws=1);
void mtp_prob_target(const strata::kernels::RadixSampler&,int,const strata::kernels::SamplerParams&,
    const MtpProbCursor*,unsigned,const MtpProbRow*,const int*,int*,unsigned*,cudaStream_t,bool has_draft,int draws=1);
}
