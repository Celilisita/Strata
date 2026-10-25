#pragma once
#include "strata/gemma4/native/q6_aligned.hpp"
#include <cstdint>

namespace strata::gemma4::native {
struct Q6DownStripeBinding {
    const void* owner=nullptr;
    std::uint64_t epoch=0;
    cudaStream_t stream=nullptr;
};
struct Q6DownStripeLease {
    AlignedQ6View weight;
    SingleQ8View input;
    Q6DownStripeBinding binding;
};
// Explicit T1 Down research primitive. Same224 payload/Single40 math, four
// independent K stripes. Caller retains all spans on its live ordered stream.
// No Model hook, allocation, synchronization, repacking or GPU fallback here.
void linear_q6_down_striped_t1(Q6DownStripeLease,float*,Q6DownStripeBinding live);
}
