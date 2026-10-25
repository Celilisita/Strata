#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35moe {
// Process-scoped TG option; the PP ATTN_SHARED_JOIN setting is independent.
inline bool tg_attention_shared_join_enabled() {
    static const bool enabled = [] {
        const char* raw = std::getenv("STRATA_QWEN35_TG_SHARED_JOIN");
        if (!raw || std::strcmp(raw, "0") == 0) return false;
        if (std::strcmp(raw, "1") == 0) return true;
        throw std::invalid_argument("TG shared join requires exactly0 or1");
    }();
    return enabled;
}
} // namespace strata::qwen35moe
