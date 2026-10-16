#pragma once
#include "ggml.h"
#include "pp_cache_view.hpp"
#include <cstdint>
#include <cstring>

namespace strata::deepseek_v4 {
inline constexpr uint64_t pp_stream_magic = 0x5354524154505031ull;
// Starts after GGML's custom3 callback parameters; pointers refer to immutable
// model tensors, which outlive every graph and the backend scratch service.
struct PpStreamSources {
    const ggml_tensor *gate, *up, *down;
    float clamp;
    uint32_t window_experts;
    const PpCacheView* cache = nullptr;
};
inline constexpr size_t pp_stream_params_offset = 24;
static_assert(sizeof(PpStreamSources) + pp_stream_params_offset <= GGML_MAX_OP_PARAMS);
inline bool is_pp_stream(const ggml_tensor* t) {
    if (!t || t->op != GGML_OP_MAP_CUSTOM3) return false;
    uint64_t tag = 0;
    // userdata is offset 16 in the 64-bit custom3 parameters. It is a tag,
    // never dereferenced and never used as a CPU inference callback context.
    std::memcpy(&tag, reinterpret_cast<const char*>(t->op_params) + 16, sizeof(tag));
    return tag == pp_stream_magic;
}
inline PpStreamSources pp_stream_sources(const ggml_tensor* t) {
    PpStreamSources s{};
    std::memcpy(&s, reinterpret_cast<const char*>(t->op_params) + pp_stream_params_offset, sizeof(s));
    return s;
}
ggml_tensor* build_pp_stream(ggml_context*, ggml_tensor* x, ggml_tensor* ids, ggml_tensor* weights,
                            ggml_tensor* gate, ggml_tensor* up, ggml_tensor* down, float clamp,
                            int window_experts = 32, const PpCacheView* cache = nullptr);
bool pp_stream_enabled();
}
