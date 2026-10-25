#pragma once
#include <cstdint>

namespace strata::kernels::cpu {
// This translation unit is compiled for AVX2/F16C. Check cpu_avx2_ok first.
float nvfp4_avx2_r4_dot(const uint8_t* weight, const void* input, int columns);
float nvfp4_avx2_v2_dot(const uint8_t* weight, const void* input, int columns);
float nvfp4_avx2_dot(const uint8_t* weight, const void* input, int columns);
}
