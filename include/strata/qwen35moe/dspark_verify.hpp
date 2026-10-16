#pragma once
#include "strata/qwen35moe/device_buffer.hpp"
#include "strata/qwen35moe/head_pick.hpp"
#include <vector>

namespace strata::qwen35moe {

// Verification belongs to Strata's target runtime.  This workspace holds
// every recurrent prefix so a rejected suffix can be rolled back precisely.
struct DsparkVerifyState {
    int capacity = 0;
    DeviceBuffer norm, logits, picks;
    std::vector<DeviceBuffer> state, history;
    std::vector<HeadPick> host_picks;
    std::vector<float> auxiliary;
};

struct DsparkVerifiedBlock {
    int next = -1;
    int verified = 0;
    std::vector<int> committed_ids;
    std::vector<float> auxiliary;
};

} // namespace strata::qwen35moe
