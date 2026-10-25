#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "../glm5_next/cold_gpu_protocol.hpp"
#include "../glm5_next/partial_down_protocol.hpp"

namespace strata::deepseek_v4 {
inline constexpr int device_cache_tag_index = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
inline constexpr int32_t device_cache_lookup_tag = 0x44434c31;
inline constexpr int32_t device_cache_guard_tag = 0x44434731;
inline constexpr int32_t device_cache_hybrid_weight_tag = 0x44434831;
inline constexpr int device_cache_max_layers = 64;
inline constexpr int device_cache_max_experts = 288;
inline constexpr std::size_t device_frequency_scores = 43 * 256;
inline constexpr std::size_t device_frequency_words = device_frequency_scores + 1;
inline constexpr int device_cache_max_routes = 32;
struct alignas(64) DeviceMissPacket {
    volatile uint32_t sequence = 0, response = 0, error = 0, layer = 0;
    int32_t ids[device_cache_max_routes]{};
    // Written only on a real miss, before sequence publication. Never read by
    // the CPU while an all-hit lookup updates the GPU frequency authority.
    uint32_t frequency[device_cache_max_layers * device_cache_max_experts + 1]{};
    // Occupies the old4-byte alignment padding; all previous field offsets stay.
    // Read only for an active16-route coldGPU hook; other requests ignore it.
    uint32_t gpu_cold_mask = 0;
    uint64_t owner_epoch = 0;
    // Native TG publishes activations only on a real miss. CPU ranks remain
    // absent from the residency directory; their contributions use this ticket.
    uint32_t native_request = 0, cpu_mask = 0, cpu_done = 0, cpu_error = 0;
    uint32_t routed_count = 0;
    float clamp = 0;
    uintptr_t gpu_slots = 0;
    int32_t gpu_order[device_cache_max_routes]{};
    int32_t reply_slots[device_cache_max_routes]{};
    float activation[4 * 4096]{};
    float contribution[device_cache_max_routes * 4096]{};
};
// Pin the pre-coldGPU64-bit packet ABI in host and CUDA compilation alike.
static_assert(sizeof(DeviceMissPacket)==664000&&alignof(DeviceMissPacket)==64,"device miss packet ABI changed");
static_assert(offsetof(DeviceMissPacket,gpu_cold_mask)==73876&&offsetof(DeviceMissPacket,owner_epoch)==73880,"cold mask escaped old padding");
static_assert(offsetof(DeviceMissPacket,native_request)==73888&&offsetof(DeviceMissPacket,routed_count)==73904,"native request ABI changed");
static_assert(offsetof(DeviceMissPacket,gpu_slots)==73912&&offsetof(DeviceMissPacket,activation)==74176&&
    offsetof(DeviceMissPacket,contribution)==139712,"device miss payload ABI changed");
inline constexpr int device_cache_counter_count = 8 + 3 * device_cache_max_layers;
inline bool device_cache_lookup(const ggml_tensor* tensor) {
    return tensor && tensor->op == GGML_OP_GET_ROWS &&
           tensor->op_params[device_cache_tag_index] == device_cache_lookup_tag;
}
inline bool device_cache_guard(const ggml_tensor* tensor) {
    return tensor && tensor->op == GGML_OP_CLAMP &&
           tensor->op_params[device_cache_tag_index] == device_cache_guard_tag;
}
inline bool device_cache_hybrid_weight(const ggml_tensor* t) {
    return t&&t->op==GGML_OP_MUL&&t->op_params[device_cache_tag_index]==device_cache_hybrid_weight_tag;
}
inline bool device_cache_supported(const ggml_tensor* tensor) {
    if(strata::glm5_next::partial_down_mid(tensor))return strata::glm5_next::partial_down_mid_supported(tensor);
    if (device_cache_lookup(tensor)) {
        if (tensor->type != GGML_TYPE_I32 || !ggml_is_contiguous(tensor) || (ggml_nelements(tensor) != 6 && ggml_nelements(tensor) != 8 && ggml_nelements(tensor) != 12 && ggml_nelements(tensor) != 16 && ggml_nelements(tensor) != 32) ||
            tensor->op_params[2] < 0 || tensor->op_params[2] >= device_cache_max_layers) return false;
        if(bool(tensor->src[8])!=bool(tensor->src[9]))return false;
        if(tensor->src[8]&&(!strata::glm5_next::partial_down_bridge_tensor(tensor->src[8])||!tensor->src[9]||
            tensor->src[9]->type!=GGML_TYPE_I32||ggml_nelements(tensor->src[9])!=8||!ggml_is_contiguous(tensor->src[9])||
            tensor->op_params[2]!=24||tensor->op_params[10]!=1||tensor->op_params[11]!=1||
            tensor->op_params[strata::glm5_next::cold_gpu_magic_word]!=strata::glm5_next::cold_gpu_op_magic||ggml_nelements(tensor)!=16))return false;
        constexpr ggml_type types[]{GGML_TYPE_I32, GGML_TYPE_I32, GGML_TYPE_I32, GGML_TYPE_I64};
        const int counts[]{int(tensor->src[0] ? ggml_nelements(tensor->src[0]) : 0), 6, 1, device_cache_counter_count};
        if (counts[0] != 256 && counts[0] != 288) return false;
        if(strata::glm5_next::cold_gpu_magic_rows(tensor->op_params[strata::glm5_next::cold_gpu_magic_word])){
            uintptr_t metadata=0;std::memcpy(&metadata,tensor->op_params+strata::glm5_next::cold_gpu_pointer_word,sizeof(metadata));
            if(!metadata||metadata%alignof(strata::glm5_next::ColdGpuLayer)||counts[0]!=288||ggml_nelements(tensor)!=16||
                !strata::glm5_next::cold_gpu_layer_valid(tensor->op_params[2])||tensor->op_params[10]!=1||tensor->op_params[11]!=1||
                !tensor->src[4]||!tensor->src[5]||!tensor->src[7]||tensor->src[5]->type!=GGML_TYPE_F32||
                ggml_nelements(tensor->src[4])<int64_t(tensor->op_params[2]+1)*288+1||tensor->src[4]->ne[1]!=1||tensor->src[4]->ne[2]!=1||tensor->src[4]->ne[3]!=1||
                tensor->src[5]->ne[0]!=4096||tensor->src[5]->ne[1]!=2||tensor->src[5]->ne[2]!=1||tensor->src[5]->ne[3]!=1||
                !ggml_is_contiguous(tensor->src[5])||tensor->src[7]->type!=GGML_TYPE_I32||
                ggml_nelements(tensor->src[7])!=2||!ggml_is_contiguous(tensor->src[7]))return false;
        }
        if(ggml_nelements(tensor)==32&&(counts[0]!=288||tensor->op_params[10]!=1||tensor->op_params[11]!=1||
            !tensor->src[4]||!tensor->src[5]||!tensor->src[7]||tensor->src[6]||
            ggml_nelements(tensor->src[4])<int64_t(tensor->op_params[2]+1)*288+1||
            tensor->src[7]->type!=GGML_TYPE_I32||ggml_nelements(tensor->src[7])!=2||!ggml_is_contiguous(tensor->src[7])||
            tensor->src[5]->type!=GGML_TYPE_F32||ggml_nelements(tensor->src[5])!=4*4096||
            !ggml_is_contiguous(tensor->src[5])))return false;
        for (int i = 0; i < 4; ++i)
            if (!tensor->src[i] || tensor->src[i]->type != types[i] || !ggml_is_contiguous(tensor->src[i]) ||
                ggml_nelements(tensor->src[i]) != (i == 1 ? ggml_nelements(tensor) : counts[i])) return false;
        if (tensor->src[4] && (tensor->src[4]->type != GGML_TYPE_I32 ||
                !ggml_is_contiguous(tensor->src[4]) || (ggml_nelements(tensor->src[4]) < 1 || ggml_nelements(tensor->src[4]) > device_cache_max_layers * device_cache_max_experts + 1 || (ggml_nelements(tensor->src[4])-1) % counts[0] != 0))) return false;
        return true;
    }
    if(device_cache_hybrid_weight(tensor)) {
        if(bool(tensor->src[6])!=bool(tensor->src[7]))return false;
        if(tensor->src[7]&&(!strata::glm5_next::partial_down_bridge_tensor(tensor->src[7])||
            tensor->src[6]->type!=GGML_TYPE_F32||ggml_nelements(tensor->src[6])!=16384||!ggml_is_contiguous(tensor->src[6])||
            tensor->op_params[2]!=24||tensor->op_params[11]!=1||tensor->ne[2]!=2))return false;
        return tensor->type==GGML_TYPE_F32&&tensor->ne[0]==4096&&tensor->ne[1]==8&&(tensor->ne[2]==1||tensor->ne[2]==2||(tensor->ne[2]==4&&tensor->ne[3]==1&&tensor->op_params[11]==1))&&
            tensor->src[0]&&tensor->src[1]&&tensor->src[2]&&tensor->src[2]->type==GGML_TYPE_I32&&
            ggml_nelements(tensor->src[2])==8*tensor->ne[2]&&ggml_nelements(tensor->src[1])==8*tensor->ne[2]&&
            ggml_is_contiguous(tensor)&&ggml_is_contiguous(tensor->src[0])&&ggml_is_contiguous(tensor->src[1])&&ggml_is_contiguous(tensor->src[2]);
    }
    if (device_cache_guard(tensor))
        return tensor->type == GGML_TYPE_F32 && tensor->src[0] && tensor->src[1] &&
            tensor->src[0]->type == GGML_TYPE_F32 && tensor->src[1]->type == GGML_TYPE_I32 &&
            ggml_nelements(tensor->src[1]) == 1 && ggml_nelements(tensor) == ggml_nelements(tensor->src[0]) &&
            ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0]) && ggml_is_contiguous(tensor->src[1]);
    return false;
}
// Counters0..4: lookup/all-hit/hit-routes/misses/timeouts;5..7 token totals/state; then3 per layer.
}
