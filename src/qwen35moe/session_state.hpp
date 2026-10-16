#pragma once
#include "strata/qwen35moe/runtime.hpp"
#include "visual_state.hpp"

namespace strata::qwen35moe {
struct SessionLayerState {
    DeviceBuffer history, state, key, value;
};
struct SessionState {
    std::vector<SessionLayerState> layers;
    DeviceBuffer head_logits, mtp_seed, mtp_key, mtp_value, verify_logits;
    HeadPick pick{-1, -1, 0};
    const float* selected_logits = nullptr;
    std::unique_ptr<VisualState> visual;
    std::vector<int> sampling_tail;
    uint64_t sampling_counter = 0;
};
// The active slot's device owners live in Runtime; parked slots own only mutable
// sequence state. Dense/expert weights, directory, staging and pool stay shared.
struct SessionSet {
    std::array<SessionState, 2> slots;
    int active = 0;
    size_t additional_bytes = 0;
    DeviceBuffer picks;
};
} // namespace strata::qwen35moe
