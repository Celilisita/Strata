#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35moe::detail {
inline bool tg_phase_lease_enabled() {
    const char* raw = std::getenv("STRATA_QWEN35_TG_PHASE_LEASE");
    if (!raw) return false;
    const std::string value(raw);
    if (value != "0" && value != "1")
        throw std::invalid_argument("TG_PHASE_LEASE must be0 or1");
    return value == "1";
}
inline bool tg_phase_verify_enabled() {
    const char* raw = std::getenv("STRATA_QWEN35_TG_PHASE_VERIFY");
    if (!raw) return false;
    const std::string value(raw);
    if (value != "0" && value != "1")
        throw std::invalid_argument("TG_PHASE_VERIFY must be0 or1");
    return value == "1";
}
} // namespace strata::qwen35moe::detail
