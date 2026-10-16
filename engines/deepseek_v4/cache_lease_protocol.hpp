#pragma once
#include "ggml.h"

namespace strata::deepseek_v4 {
inline constexpr int lease_tag_index = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
inline constexpr int32_t lease_lookup_tag = 0x4c534c31;
inline constexpr int32_t lease_guard_tag = 0x4c534731;
inline bool lease_lookup(const ggml_tensor* t) {
    return t && t->op == GGML_OP_GET_ROWS && t->op_params[lease_tag_index] == lease_lookup_tag;
}
inline bool lease_guard(const ggml_tensor* t) {
    return t && t->op == GGML_OP_CLAMP && t->op_params[lease_tag_index] == lease_guard_tag;
}
}
