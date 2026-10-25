#pragma once
#include "strata/kernels/sampler.hpp"
#include <memory>

namespace strata::kernels {
struct SamplePick {
    int id = -1, first_bad = -1;
    float logit = 0, probability = 0;
};
// Reusable GPU workspace; all vocabulary transforms stay on the device.
// Stable radix sorting preserves the lower-ID tie rule and supports top_k=0
// (all vocabulary), without a hidden 64-token truncation.
class RadixSampler {
public:
    explicit RadixSampler(int vocabulary);
    ~RadixSampler();
    RadixSampler(const RadixSampler&) = delete;
    RadixSampler& operator=(const RadixSampler&) = delete;
    void sample(const float* logits, const int* history, int history_length,
        const SamplerParams& params, void* stream, const int* failed = nullptr);
    // Same chain with a device draw index, for captured speculative windows.
    void sample_counter(const float* logits, const int* history, int history_length,
        const SamplerParams& params, void* stream, const uint64_t* counter,
        unsigned offset, const int* failed = nullptr);
    void write_pick(int* id, unsigned* invalid, void* stream) const;
    const SamplePick* result() const;
    size_t workspace_bytes() const;
    // Bounded benchmark interface: sorted support and unnormalized final CDF.
    const int* sorted_ids() const;
    const double* cumulative() const;
    const int* kept_count() const;
    static void validate(const SamplerParams& params);
private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace strata::kernels
