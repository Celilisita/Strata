#pragma once
#include "strata/kernels/radix_sampler.hpp"
#include "strata/models/request_sampling.hpp"
#include "strata/runtime/capacity_storage.hpp"
#include <cuda_runtime.h>
#include <vector>
namespace strata::gemma4::native {
// Request-owned sampling state. Greedy-only draft verification does not consume
// it; positive temperature uses the ordinary complete GPU sampler chain.
class ModelSampling {
    kernels::RadixSampler sampler_;
    models::RequestSampling request_;
    runtime::CapacityStorage storage_;
    int* history_ = nullptr;
public:
    ModelSampling(int vocabulary, const kernels::SamplerParams&);
    bool active() const { return request_.enabled(); }
    void begin(const std::vector<int>& prompt) { request_.prepare(prompt); }
    void enqueue(float* logits, int vocabulary, float cap, int* next,
                 unsigned* invalid, cudaStream_t);
    void observe(int token) {
        request_.remember(token);
        ++request_.params.counter;
    }
};
void finish_sampling_pick(const kernels::SamplePick*, int*, unsigned*, cudaStream_t);
void prepare_sampling_logits(float*, int vocabulary, float cap, unsigned*, cudaStream_t);
} // namespace strata::gemma4::native
