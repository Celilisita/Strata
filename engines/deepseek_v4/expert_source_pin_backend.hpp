#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::deepseek_v4 {

// Called after the selected backend's event constructor sets the CUDA device.
// Registration and release never depend on a mutable process environment.
bool register_readonly_source(void* data, std::size_t bytes, uint32_t& error) noexcept;
void unregister_readonly_source(void* data) noexcept;

} // namespace strata::deepseek_v4
