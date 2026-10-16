#pragma once
#include "cache_lease_protocol.hpp"
#include <cstdint>
#include <cstring>

namespace strata::deepseek_v4 {
// Host metadata only. Each key is a stable address owned by the cache lease,
// distinct for batches1..8. Canonical CUDA graph property checks still decide
// whether a graph can replay; this only avoids PP/TG overwriting the same entry.
inline void set_lease_graph_key(ggml_tensor* guard, const void* key) {
    static_assert(2 * sizeof(int32_t) + sizeof(uintptr_t) < lease_tag_index * sizeof(int32_t));
    const auto value = reinterpret_cast<uintptr_t>(key);
    std::memcpy(guard->op_params + 2, &value, sizeof(value));
}
inline const void* lease_graph_key(const ggml_tensor* tail) {
    if (!lease_guard(tail)) return nullptr;
    uintptr_t value = 0;
    std::memcpy(&value, tail->op_params + 2, sizeof(value));
    return reinterpret_cast<const void*>(value);
}
}
