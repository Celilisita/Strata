#pragma once
// Explicit fixture ABI only. The caller owns the immutable private host view;
// the returned buffer owns registration and must die before that view.
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>

namespace strata::glm5_next {
struct MappedSourceInfo {
    void* device_base = nullptr;
    uint64_t registered_bytes = 0;
    uint32_t flags = 0, cuda_error = 0;
    int can_map = 0, unified = 0, host_pointer_supported = 0, device_flags = 0, l2_bytes = 0;
};
// device_base aliases the exact supplied host registration start. It need not
// equal that host pointer. Source page/range/COW checks belong to the caller.
using ProbeMapSource = ggml_backend_buffer_t (*)(ggml_backend_t, void*, size_t, MappedSourceInfo*);
}
