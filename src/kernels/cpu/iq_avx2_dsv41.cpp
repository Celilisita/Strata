// include/strata/kernels/cpu/iq_avx2.hpp's DeepSeek-V4.1 AVX2 rows.  Upstream's implementation lives in
// src/kernels/cpu/iq_avx2.cpp, whose helpers this tree does not have; the lane asks iq256x_supported(), gets
// "no", and uses its AVX-512 rows (iq512x_rows) or ggml-cpu's own vec_dot instead.  Same bits either way.
#include "strata/kernels/cpu/iq_avx2.hpp"

namespace strata::kernels::cpu {

bool iq256x_supported(int) noexcept { return false; }

void iq256x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}

}  // namespace strata::kernels::cpu
