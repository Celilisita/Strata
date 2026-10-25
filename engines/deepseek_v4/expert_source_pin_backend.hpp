#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::deepseek_v4 {

// Called after the selected backend's event constructor sets the CUDA device.
// Registration and release never depend on a mutable process environment.
struct SourcePinBackendInfo { bool cuda_compiled; uint32_t flags; };
// Also used by bounded source-mapping diagnostics; no registration or page touch.
SourcePinBackendInfo source_pin_backend_info(const void* data, std::size_t bytes) noexcept;
bool register_readonly_source(void* data, std::size_t bytes, uint32_t& error, bool mapped = false) noexcept;
void unregister_readonly_source(void* data) noexcept;
// Raw API return and the separate thread-local last-error slot never share a channel.
struct SourcePinApiResult { uint32_t primary=0, pending=0; const char* stage="api"; };
// The backend event constructor selected this captured device. Never switch owners.
bool synchronize_source_registration(int device, SourcePinApiResult& result) noexcept;
bool unregister_source_checked(int device, void* data, SourcePinApiResult& result) noexcept;
bool register_source_checked(int device, void* data, std::size_t bytes, bool mapped, SourcePinApiResult& result) noexcept;
uintptr_t source_alias_checked(int device, void* data, SourcePinApiResult& result) noexcept;
// Consume only the exact current OOM2 or register-NotSupported801 already recorded.
bool clear_source_registration_error(int device, uint32_t expected, uint32_t& observed) noexcept;
// Soft WDDM budget of this process on the selected adapter, not physical free memory.
struct SourceAliasBudget {
    uint64_t budget=0, usage=0;
    uint32_t node=0, luid_low=0, luid_high=0, error=0;
    bool required=false;
    const char* stage="not_required";
};
bool query_source_alias_budget(int device, SourceAliasBudget& info) noexcept;
bool selected_source_cuda_uuid(int device, char* output, std::size_t capacity, uint32_t& error) noexcept;
int source_alias_current_device() noexcept;
uintptr_t registered_source_alias(void* data, uint32_t& error) noexcept;
bool drain_source_alias_readers(int device, uint32_t& error) noexcept;

} // namespace strata::deepseek_v4
