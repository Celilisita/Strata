#include "sampling_positioned.hpp"
#include "strata/kernels/radix_sampler.hpp"
#include "strata/kernels/sampler_math.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::completed_v100::qwen35::native {
using strata::kernels::SamplerParams;
using strata::kernels::SamplePick;
namespace sampler_math = strata::kernels::sampler_math;
namespace {
__global__ void positioned_draw(const float* logits,const int* ids,const double* cdf,
    const int* kept,SamplerParams params,const int* position,int origin,int row,
    SamplePick* result) {
    if(threadIdx.x||blockIdx.x)return;
    const int index=*position-origin+row;
    if(index<0){result->id=-1;result->first_bad=-3;return;}
    const double total=cdf[*kept-1];int chosen=0;
    if(!params.greedy&&params.temperature>0&&isfinite(total)&&total>0) {
        const double target=double(sampler_math::uniform(params.seed,params.counter+uint64_t(index)))*total;
        int low=0,high=*kept;
        while(low<high){const int mid=low+(high-low)/2;
            if(cdf[mid]<=target)low=mid+1;else high=mid;}
        chosen=min(low,*kept-1);
    }
    const int id=ids[chosen];
    const double mass=cdf[chosen]-(chosen?cdf[chosen-1]:0.0);
    // Finite diagnostics were established by the unchanged ordinary sampler.
    result->id=id;result->logit=logits[id];result->probability=float(mass/total);
}
}
void sample_positioned(strata::kernels::RadixSampler& sampler,const float* logits,const int* history,int length,
    const SamplerParams& params,const int* position,int origin,int row,void* stream,const int* failed) {
    if(!position||origin<0||row<0||row>2)throw std::invalid_argument("Invalid GPU sample cursor");
    sampler.sample(logits,history,length,params,stream,failed);
    positioned_draw<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(logits,sampler.sorted_ids(),sampler.cumulative(),
        sampler.kept_count(),params,position,origin,row,
        const_cast<SamplePick*>(sampler.result()));
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("GPU positioned sampler launch failed");
}
}
