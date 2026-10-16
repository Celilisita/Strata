#pragma once

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace strata::qwen35moe {

struct CpuExpertState {
    explicit CpuExpertState(int first_layer) : from_layer(first_layer) {}

    kernels::cpu::ExpertPool pool;
    int from_layer;
    std::vector<float> norm, parts;
    std::vector<uint8_t> activations;
    std::array<kernels::cpu::ExpertJobMulti, 8> jobs{};
};

} // namespace strata::qwen35moe
