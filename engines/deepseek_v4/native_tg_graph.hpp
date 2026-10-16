#pragma once
#include "device_cache_protocol.hpp"
#include <cstring>
namespace strata::deepseek_v4 {
inline constexpr uint64_t native_tg_magic = 0x5354524154544731ull;
struct NativeTgSources {
    const ggml_tensor *gate, *up, *down;
    uintptr_t packet;
    float clamp;
    uint16_t layer, reserved;
};
static_assert(sizeof(NativeTgSources) == 40);
inline bool is_native_tg(const ggml_tensor* t) {
    if (!t || t->op != GGML_OP_MAP_CUSTOM3) return false;
    uint64_t tag = 0;
    std::memcpy(&tag, reinterpret_cast<const char*>(t->op_params) + 16, sizeof(tag));
    return tag == native_tg_magic;
}
inline NativeTgSources native_tg_sources(const ggml_tensor* t) {
    NativeTgSources s{};
    std::memcpy(&s, reinterpret_cast<const char*>(t->op_params) + 24, sizeof(s));
    return s;
}
bool native_tg_enabled();
bool native_tg_cpu_only();
ggml_tensor* finish_native_tg(ggml_context*, ggml_tensor* start, ggml_tensor* shared);
ggml_tensor* build_native_tg(ggml_context*, ggml_tensor* x, ggml_tensor* ids, ggml_tensor* weights,
    ggml_tensor* slots, ggml_tensor* gate_bank, ggml_tensor* up_bank, ggml_tensor* down_bank,
    ggml_tensor* gate_source, ggml_tensor* up_source, ggml_tensor* down_source, float clamp, int layer);
}
