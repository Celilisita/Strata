#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <chrono>

namespace strata::deepseek_v4 {
template<class Plan> bool timed_cache_plan(Plan plan, uint64_t* elapsed_ns) {
    if (!elapsed_ns) return plan();
    const auto start = std::chrono::steady_clock::now();
    const bool result = plan();
    *elapsed_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
    return result;
}
// Both scheduler copies and the miss worker use the canonical cache's fill list.
template<class Weights, class Banks, class Fills, class Upload>
size_t upload_cache_fills(const Weights& weights, const Banks& banks,
                         const Fills& fills, Upload upload) {
    size_t bytes = 0;
    for (size_t role = 0; role < weights.size(); ++role) {
        const auto* weight = weights[role];
        const size_t stride = weight->nb[2];
        for (size_t i = 0; i < fills.size();) {
            size_t count = 1;
            while (i + count < fills.size() &&
                   fills[i + count].expert == fills[i].expert + int32_t(count) &&
                   fills[i + count].slot == fills[i].slot + int32_t(count)) ++count;
            upload(banks[role], static_cast<const uint8_t*>(weight->data) + fills[i].expert * stride,
                   fills[i].slot * stride, count * stride);
            bytes += count * stride;
            i += count;
        }
    }
    return bytes;
}
}
