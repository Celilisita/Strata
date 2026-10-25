#pragma once
#include <cstring>
#include <stdexcept>

namespace strata::prefill::mmq {
// Dense-only, per-format opt-in. Routed providers keep their selection.
inline unsigned blackwell_dense_int8_mask(const char* value) {
    if (!value || !std::strcmp(value, "0")) return 0;
    if (!std::strcmp(value, "12")) return 1u << 12;
    if (!std::strcmp(value, "13")) return 1u << 13;
    if (!std::strcmp(value, "12,13")) return (1u << 12) | (1u << 13);
    throw std::invalid_argument("dense INT8 requires0,12,13 or12,13");
}
inline bool blackwell_dense_int8_selected(unsigned mask, int type, int rows, int width, int outputs) {
    const bool domain = type == 12 ? ((width == 2048 && (outputs == 4096 || outputs == 8192)) ||
        (width == 4096 && outputs == 2048)) : type == 13 && width == 2048 && outputs == 8192;
    return rows >= 1024 && rows <= 2048 && domain && (mask & (1u << type));
}
} // namespace strata::prefill::mmq
