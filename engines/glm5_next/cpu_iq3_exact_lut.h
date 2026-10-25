#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
enum { STRATA_IQ3_LUT_CHUNK_ROWS = 128, STRATA_IQ3_LUT_TILE_ROWS = 32 };
// Setup-only:bit0 actual compiledAVX2 body, bit1 CPUAVX2/FMA/F16C,
// bit2 OS XMM/YMM state. VNNI/AVX512 are not required by this implementation.
unsigned strata_iq3_exact_lut_capabilities(void);
// Setup-only compile-time table versus all512 original grid entries.
bool strata_iq3_exact_lut_grid_identity(void);
// Eight-argument custom leaf, never cast to ggml_vec_dot_t. Original rows are
//128 consecutive IQ3_S outputrows with1760-byte stride and4096 inputs.
// Ny1 hasout1/q8b bothNULL; Ny2 hasboth nonNULL. Outputs own128 floats each.
// Q8_K inputs are the unchanged canonical rows in[-127,127]. Every call builds
// exactint16 LUTs inside its16blockouter loop and shares each across128rows.
// False performs no output writes. No weight prepack/allocation/persistent view.
bool strata_iq3_exact_lut_chunk(int n, float * out0, float * out1,
    const void * original, size_t row_stride, const void * q8a, const void * q8b, size_t rows);
#ifdef __cplusplus
}
#endif
