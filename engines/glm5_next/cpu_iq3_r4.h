#pragma once
// Exact diagnostic IQ3_S R4 layout. Original tensors remain authoritative.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { STRATA_IQ3_R4_QK = 256, STRATA_IQ3_R4_BLOCK_BYTES = 440 };
struct strata_iq3_r4_block {
    uint16_t d[4];
    uint8_t qs[256], qh[32], signs[128], scales[16];
};
struct strata_iq3_r4_rows {
    const void * data; // One group of four output rows, blocks contiguous R4 blocks.
    size_t blocks;
};

// No allocation or requantization. rows is a positive multiple of four,
// at most2048; row_blocks is1..16. The output size equals rows*row_blocks*110.
bool strata_iq3_r4_payload_bytes(size_t rows, size_t row_blocks, size_t * bytes);
bool strata_iq3_r4_pack(const void * original, size_t rows, size_t row_blocks,
                      void * packed, size_t packed_bytes);
// Setup-only inverse permutation checks every original d/qs/qh/sign/scale byte.
bool strata_iq3_r4_identity(const void * original, size_t rows, size_t row_blocks,
                          const void * packed, size_t packed_bytes);

// Setup-only capability check:7 means the existing AVX2/VNNI256 backend and
// OS state are available. The caller checks this before any dot invocation.
unsigned strata_iq3_r4_capabilities(void);
// Each output owns four rows. Strides are bytes, normally sizeof(float).
// Ny1:out1 and q8b are bothNULL. Ny2:both are nonNULL. On false no output writes.
// The exact custom eight-argument prototype is never cast to ggml_vec_dot_t.
typedef bool (*strata_iq3_r4_dot_fn)(int, float *, size_t, float *, size_t,
                                  struct strata_iq3_r4_rows, const void *, const void *);
bool strata_iq3_r4_dot(int n, float * out0, size_t stride0, float * out1, size_t stride1,
                     struct strata_iq3_r4_rows rows, const void * q8a, const void * q8b);

#ifdef __cplusplus
}
#endif
