#include "expert_source_pin_backend.hpp"

#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
#include <cuda_runtime_api.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#endif

namespace strata::deepseek_v4 {

bool register_readonly_source(void* data, std::size_t bytes, uint32_t& error) noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    unsigned flags = cudaHostRegisterPortable | cudaHostRegisterReadOnly;
#ifdef _WIN32
    // Only an explicitly private file view may omit ReadOnly. Validate the
    // entire range, retaining the read-only flag for other mappings.
    auto* cursor = static_cast<const char*>(data);
    std::size_t left = bytes;
    bool private_source = true;
    while (left) {
        MEMORY_BASIC_INFORMATION region{};
        if (!VirtualQuery(cursor, &region, sizeof(region)) || region.State != MEM_COMMIT ||
            region.Type != MEM_MAPPED || region.Protect != PAGE_WRITECOPY) {
            private_source = false;
            break;
        }
        const auto available = static_cast<std::size_t>(
            static_cast<const char*>(region.BaseAddress) + region.RegionSize - cursor);
        const auto step = available < left ? available : left;
        if (!step) { private_source = false; break; }
        cursor += step;
        left -= step;
    }
    if (private_source) flags = cudaHostRegisterPortable;
#endif
    const auto result = cudaHostRegister(data, bytes, flags);
    if (result == cudaSuccess) return true;
    error = static_cast<uint32_t>(result);
    (void) cudaGetLastError();
#else
    (void) data;
    (void) bytes;
    error = 801; // cudaErrorNotSupported; CPU builds cannot register CUDA sources.
#endif
    return false;
}

void unregister_readonly_source(void* data) noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    if (cudaHostUnregister(data) != cudaSuccess) (void) cudaGetLastError();
#else
    (void) data;
#endif
}

} // namespace strata::deepseek_v4
