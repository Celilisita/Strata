#pragma once
#include <stdexcept>
#include <string>

namespace strata::qwen35moe {
inline unsigned precise_dense_types(const std::string& csv) {
    if (csv == "0") return 0;
    if (csv == "12") return 1u << 12;
    if (csv == "13") return 1u << 13;
    if (csv == "14") return 1u << 14;
    if (csv == "12,13") return (1u << 12) | (1u << 13);
    if (csv == "12,14") return (1u << 12) | (1u << 14);
    if (csv == "13,14") return (1u << 13) | (1u << 14);
    if (csv == "12,13,14") return (1u << 12) | (1u << 13) | (1u << 14);
    throw std::invalid_argument("PRECISE_DENSE requires0 or sorted unique subset of12,13,14");
}
} // namespace strata::qwen35moe
