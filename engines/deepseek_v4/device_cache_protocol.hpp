#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>

namespace strata::deepseek_v4 {
inline constexpr int device_cache_tag_index = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
inline constexpr int32_t device_cache_lookup_tag = 0x44434c31;
inline constexpr int32_t device_cache_guard_tag = 0x44434731;
inline constexpr int device_cache_max_layers = 44;
inline constexpr std::size_t device_frequency_scores = 43 * 256;
inline constexpr std::size_t device_frequency_words = device_frequency_scores + 1;
struct alignas(64) DeviceMissPacket {
    volatile uint32_t sequence = 0, response = 0, error = 0, layer = 0;
    int32_t ids[12]{};
    // Written only on a real miss, before sequence publication. Never read by
    // the CPU while an all-hit lookup updates the GPU frequency authority.
    uint32_t frequency[device_cache_max_layers * 256 + 1]{};
    // Native TG publishes activations only on a real miss. CPU ranks remain
    // absent from the residency directory; their contributions use this ticket.
    uint32_t native_request = 0, cpu_mask = 0, cpu_done = 0, cpu_error = 0;
    float clamp = 0;
    uintptr_t gpu_slots = 0;
    int32_t gpu_order[12]{};
    int32_t reply_slots[12]{};
    float activation[2 * 4096]{};
    float contribution[12 * 4096]{};
};
inline bool device_cache_lookup(const ggml_tensor* tensor) {
    return tensor && tensor->op == GGML_OP_GET_ROWS &&
           tensor->op_params[device_cache_tag_index] == device_cache_lookup_tag;
}
inline bool device_cache_guard(const ggml_tensor* tensor) {
    return tensor && tensor->op == GGML_OP_CLAMP &&
           tensor->op_params[device_cache_tag_index] == device_cache_guard_tag;
}
inline bool device_cache_supported(const ggml_tensor* tensor) {
    if (device_cache_lookup(tensor)) {
        if (tensor->type != GGML_TYPE_I32 || !ggml_is_contiguous(tensor) || (ggml_nelements(tensor) != 6 && ggml_nelements(tensor) != 12) ||
            tensor->op_params[2] < 0 || tensor->op_params[2] >= device_cache_max_layers) return false;
        constexpr ggml_type types[]{GGML_TYPE_I32, GGML_TYPE_I32, GGML_TYPE_I32, GGML_TYPE_I64};
        constexpr int counts[]{256, 6, 1, 5};
        for (int i = 0; i < 4; ++i)
            if (!tensor->src[i] || tensor->src[i]->type != types[i] || !ggml_is_contiguous(tensor->src[i]) ||
                ggml_nelements(tensor->src[i]) != (i == 1 ? ggml_nelements(tensor) : counts[i])) return false;
        if (tensor->src[4] && (tensor->src[4]->type != GGML_TYPE_I32 ||
                !ggml_is_contiguous(tensor->src[4]) || (ggml_nelements(tensor->src[4]) != device_frequency_words && ggml_nelements(tensor->src[4]) != device_cache_max_layers * 256 + 1))) return false;
        return true;
    }
    if (device_cache_guard(tensor))
        return tensor->type == GGML_TYPE_F32 && tensor->src[0] && tensor->src[1] &&
            tensor->src[0]->type == GGML_TYPE_F32 && tensor->src[1]->type == GGML_TYPE_I32 &&
            ggml_nelements(tensor->src[1]) == 1 && ggml_nelements(tensor) == ggml_nelements(tensor->src[0]) &&
            ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0]) && ggml_is_contiguous(tensor->src[1]);
    return false;
}
// Counters: lookups, all-hit lookups, hit routes, miss requests, timeouts.
inline constexpr int device_cache_counter_count = 5;
}
