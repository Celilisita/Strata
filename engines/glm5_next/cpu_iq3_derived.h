#pragma once
// Diagnostic-only exact IQ3_S representation. Original quant banks remain
// authoritative; the caller owns/retires these three borrowed SoA spans.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    STRATA_IQ3_DERIVED_QK = 256,
    STRATA_IQ3_ORIGINAL_BLOCK_BYTES = 110,
    STRATA_IQ3_DERIVED_BLOCK_BYTES = 262
};

struct strata_iq3_derived_row {
    const int8_t * levels;          // blocks*256 signed original codebook levels
    const uint16_t * d_bits;        // blocks unchanged original FP16 bit patterns
    const uint8_t * packed_scales;  // blocks*4 unchanged original packed nibbles
    size_t blocks;
};

// No allocation or float weight conversion. A role/expert has 32768 blocks;
// one 4096-input output row has 16 blocks. All destinations must be disjoint
// from each other and from the original source. False performs no writes.
bool strata_iq3_derived_payload_bytes(size_t blocks, size_t * bytes);
bool strata_iq3_derived_predecode(const void * original, size_t blocks,
    int8_t * levels, uint16_t * d_bits, uint8_t * packed_scales);
// Exact setup-only identity check. It compares all levels/d/scales against
// original IQ3_S without materializing F32 weights or changing either input.
bool strata_iq3_derived_identity(const void * original, struct strata_iq3_derived_row row);

// This is build availability, not CPUID. The existing AVX2 CPU backend and
// unchanged canonical Q8_K quantizer must already be selected by the caller.
int strata_iq3_derived_compiled_avx2(void);
// Q8 pointers borrow canonical block_q8_K rows. Their quantization, scale and
// bsums storage remain unchanged. On false no output is modified; caller may
// use the original row with the existing canonical single/pair fallback.
bool strata_iq3_derived_dot_single(int n, float * out,
    struct strata_iq3_derived_row row, const void * canonical_q8);
bool strata_iq3_derived_dot_pair(int n, float * out0, float * out1,
    struct strata_iq3_derived_row row, const void * canonical_q8a,
    const void * canonical_q8b);

#ifdef __cplusplus
}
#endif
