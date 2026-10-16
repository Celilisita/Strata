#pragma once
#include "strata/qwen35moe/runtime.hpp"
#include "strata/kernels/radix_sampler.hpp"

namespace strata::qwen35moe {
struct SamplingState {
    kernels::SamplerParams initial, current;
    kernels::RadixSampler gpu;
    DeviceBuffer history;
    std::vector<int> tail, saved_tail;
    std::array<std::vector<int>, 2> upload_rows;
    uint64_t saved_counter = 0;
    kernels::SamplePick pick;
    bool verify = false;
    int verified = 0;
    SamplingState(int vocabulary, const kernels::SamplerParams& p)
        : initial(p), current(p), gpu(vocabulary), history(size_t(p.penalty_last_n) * sizeof(int)) {
        tail.reserve(size_t(p.penalty_last_n)); saved_tail.reserve(size_t(p.penalty_last_n));
        for (auto& row : upload_rows) row.reserve(size_t(p.penalty_last_n));
    }
};
} // namespace strata::qwen35moe
