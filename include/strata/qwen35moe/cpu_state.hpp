#pragma once

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace strata::qwen35moe {

inline int configured_cpu_workers() {
    const char* raw = std::getenv("STRATA_QWEN35_CPU_WORKERS");
    if (!raw) return 0;
    int workers = 0;
    for (const char* p = raw; *p; ++p) {
        if (*p < '0' || *p > '9')
            throw std::invalid_argument("CPU_WORKERS must be an integer in 1..64");
        workers = workers * 10 + (*p - '0');
        if (workers > 64)
            throw std::invalid_argument("CPU_WORKERS must be an integer in 1..64");
    }
    if (workers < 1)
        throw std::invalid_argument("CPU_WORKERS must be an integer in 1..64");
    return workers;
}

struct CpuExpertState {
    // The host can also execute jobs; N workers therefore allow N + 1 participants.
    explicit CpuExpertState(int first_layer)
        : pool(configured_cpu_workers()), from_layer(first_layer) {}

    kernels::cpu::ExpertPool pool;
    int from_layer;
    std::vector<float> norm, parts;
    std::vector<uint8_t> activations;
    std::array<kernels::cpu::ExpertJobMulti, 8> jobs{};
};

} // namespace strata::qwen35moe
