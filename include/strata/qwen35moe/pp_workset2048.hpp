#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::qwen35moe::detail {
inline bool pp_workset2048_enabled() {
    const char* raw = std::getenv("STRATA_QWEN35_PP_WORKSET_2048");
    if (!raw || std::strcmp(raw, "0") == 0) return false;
    if (std::strcmp(raw, "1") != 0)
        throw std::invalid_argument("PP_WORKSET_2048 requires 0 or 1");
    return true;
}

inline int pp_workset2048_capacity(int original, int tokens) {
    // The configured limit is an upper bound. Short requests retain their
    // original single-batch entry; long requests reuse run_prefill_long().
    return pp_workset2048_enabled() && tokens > 2048 && original > 2048
        ? 2048 : original;
}

inline size_t pp_workset2048_reserve(int kv, bool cold_gemm) {
    if (kv < 1 || kv > 196608)
        throw std::invalid_argument("invalid PP_WORKSET_2048 context bound");
    // Existing planned upper bound: owned batch + bounded provider workspace.
    // Qualification must check physical TC + MMQ pool <= 28,549,120 bytes.
    constexpr size_t planned = size_t(2048) * 172032 + (size_t(32) << 20);
    // Full residual is at most kv rows, including the new long entry for 8K.
    // A live Gemm's explicit 64 MiB is already part of occupied_bytes.
    return planned + size_t(kv) * 8192 + (size_t(128) << 20) +
        (cold_gemm ? (size_t(64) << 20) : 0);
}
} // namespace strata::qwen35moe::detail
