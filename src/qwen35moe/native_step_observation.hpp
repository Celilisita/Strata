#pragma once
#include "strata/qwen35moe/metrics.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::qwen35moe {
// Existing STEP response only: one printf, no extra device work/readback/log.
// The caller evaluates elapsed_ms at the original response timing boundary.
inline void print_native_step(int next, double milliseconds, const Metrics& m) {
    static const bool observe = [] {
        const char* value = std::getenv("STRATA_NATIVE_STEP_METRICS");
        if (!value || !std::strcmp(value, "0")) return false;
        if (!std::strcmp(value, "1")) return true;
        throw std::invalid_argument("STRATA_NATIVE_STEP_METRICS must be0 or1");
    }();
    if (!observe) {
        std::printf("STEPPED next=%d ms=%.3f\n", next, milliseconds);
        return;
    }
    static const bool boundaries = [] {
        const char* value = std::getenv("STRATA_NATIVE_STEP_BOUNDARIES");
        if (!value || !std::strcmp(value, "0")) return false;
        if (!std::strcmp(value, "1")) return true;
        throw std::invalid_argument("STRATA_NATIVE_STEP_BOUNDARIES must be0 or1");
    }();
    if (boundaries) {
        std::printf("STEPPED next=%d ms=%.3f obs_schema=2 "
            "obs_h2d=%llu obs_hits=%llu obs_launches=%llu obs_syncs=%llu obs_jobs=%llu "
            "obs_pool_ms=%.6f obs_read=%llu obs_result=%llu obs_ids=%llu obs_head_ms=%.6f "
            "obs_cpu_layers=%llu obs_gpu_layers=%llu "
            "obs_enqueue_ms=%.6f obs_head_observe_ms=%.6f obs_registered=%llu obs_pageable=%llu\n",
            next, milliseconds,
            static_cast<unsigned long long>(m.expert_h2d),
            static_cast<unsigned long long>(m.expert_cache_hits),
            static_cast<unsigned long long>(m.expert_launches),
            static_cast<unsigned long long>(m.explicit_syncs),
            static_cast<unsigned long long>(m.cpu_jobs), m.cpu_pool_ms,
            static_cast<unsigned long long>(m.cpu_expert_read),
            static_cast<unsigned long long>(m.cpu_result_h2d),
            static_cast<unsigned long long>(m.route_d2h), m.head_wait_ms,
            static_cast<unsigned long long>(m.tg_cpu_layers),
            static_cast<unsigned long long>(m.tg_gpu_hit_layers),
            m.expert_enqueue_ms, m.head_observe_ms,
            static_cast<unsigned long long>(m.expert_h2d_registered_bytes),
            static_cast<unsigned long long>(m.expert_h2d_pageable_bytes));
        return;
    }
    std::printf("STEPPED next=%d ms=%.3f obs_schema=1 "
        "obs_h2d=%llu obs_hits=%llu obs_launches=%llu obs_syncs=%llu obs_jobs=%llu "
        "obs_pool_ms=%.6f obs_read=%llu obs_result=%llu obs_ids=%llu obs_head_ms=%.6f "
        "obs_cpu_layers=%llu obs_gpu_layers=%llu\n", next, milliseconds,
        static_cast<unsigned long long>(m.expert_h2d),
        static_cast<unsigned long long>(m.expert_cache_hits),
        static_cast<unsigned long long>(m.expert_launches),
        static_cast<unsigned long long>(m.explicit_syncs),
        static_cast<unsigned long long>(m.cpu_jobs), m.cpu_pool_ms,
        static_cast<unsigned long long>(m.cpu_expert_read),
        static_cast<unsigned long long>(m.cpu_result_h2d),
        static_cast<unsigned long long>(m.route_d2h), m.head_wait_ms,
        static_cast<unsigned long long>(m.tg_cpu_layers),
        static_cast<unsigned long long>(m.tg_gpu_hit_layers));
}
} // namespace strata::qwen35moe
