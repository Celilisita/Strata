#pragma once
#include "strata/qwen35/completed_v100/native/weights.hpp"
#include "strata/kernels/radix_sampler.hpp"

namespace strata::completed_v100::qwen35::native {
// One request's sampler state. All transforms, history assembly, draws and
// candidate IDs remain on GPU, with one workspace reused across ordered rows.
class MtpSampling {
public:
    explicit MtpSampling(int vocabulary):sampler_(vocabulary),vocabulary_(vocabulary){}
    void configure(strata::kernels::SamplerParams,int* history,int prompt);
    void pick(const float* logits,const int* inputs,int row,int position,
              const int* cursor,int* output,unsigned* invalid,cudaStream_t);
    size_t commit(const int* inputs,int count,int position,cudaStream_t);
private:
    strata::kernels::RadixSampler sampler_;
    strata::kernels::SamplerParams params_{};
    DeviceStorage storage_;
    int vocabulary_,prompt_=0,history_length_=0,capacity_=0;
    int* history_=nullptr;
    int* window_=nullptr;
};
}
