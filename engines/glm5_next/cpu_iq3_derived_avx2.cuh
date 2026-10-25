#pragma once
// Include only in the existing x86 quants.c after its canonical helpers.
// Reuse the authoritative block_iq3_s, iq3s_grid, FP16 conversion and hsum.
#include "cpu_iq3_derived.h"
#include <string.h>

static_assert(QK_K == STRATA_IQ3_DERIVED_QK, "derived IQ3 block geometry differs");
static_assert(sizeof(block_iq3_s) == STRATA_IQ3_ORIGINAL_BLOCK_BYTES,
              "derived IQ3 original storage differs");

bool strata_iq3_derived_payload_bytes(size_t blocks, size_t * bytes) {
    if (!bytes || !blocks || blocks > SIZE_MAX/STRATA_IQ3_DERIVED_BLOCK_BYTES) return false;
    *bytes = blocks*STRATA_IQ3_DERIVED_BLOCK_BYTES;
    return true;
}

static bool strata_iq3_derived_span(const void * pointer, size_t bytes,
                                  uintptr_t * begin, uintptr_t * end) {
    *begin = (uintptr_t) pointer;
    if (!pointer || !bytes || *begin > UINTPTR_MAX-bytes) return false;
    *end = *begin+bytes;
    return true;
}

static int8_t strata_iq3_derived_level(const block_iq3_s * block, int position) {
    const int group = position/32, within = position%32;
    const int tuple = within/8, lane = within%8;
    const int half = lane/4;
    const unsigned index = block->qs[group*8+2*tuple+half] |
        (((unsigned) block->qh[group] << (8-2*tuple-half)) & 256u);
    const int magnitude = (int) ((iq3s_grid[index] >> (8*(lane%4))) & 255u);
    return (int8_t) ((block->signs[group*4+tuple] & (1u << lane)) ? -magnitude : magnitude);
}

bool strata_iq3_derived_predecode(const void * original, size_t blocks,
        int8_t * levels, uint16_t * d_bits, uint8_t * packed_scales) {
    size_t bytes;
    if (!strata_iq3_derived_payload_bytes(blocks, &bytes) ||
        ((uintptr_t) original & 1u) || ((uintptr_t) d_bits & 1u)) return false;
    uintptr_t begin[4], end[4];
    if (!strata_iq3_derived_span(original, blocks*sizeof(block_iq3_s), begin, end) ||
        !strata_iq3_derived_span(levels, blocks*256, begin+1, end+1) ||
        !strata_iq3_derived_span(d_bits, blocks*2, begin+2, end+2) ||
        !strata_iq3_derived_span(packed_scales, blocks*4, begin+3, end+3)) return false;
    for (int a=0; a<4; ++a) for (int b=a+1; b<4; ++b)
        if (begin[a]<end[b] && begin[b]<end[a]) return false;
    const block_iq3_s * source = (const block_iq3_s *) original;
    for (size_t block=0; block<blocks; ++block) {
        memcpy(d_bits+block, &source[block].d, 2);
        memcpy(packed_scales+block*4, source[block].scales, 4);
        for (int lane=0; lane<256; ++lane)
            levels[block*256+lane] = strata_iq3_derived_level(source+block, lane);
    }
    return true;
}

bool strata_iq3_derived_identity(const void * original, struct strata_iq3_derived_row row) {
    size_t bytes;
    if (!original || !row.levels || !row.d_bits || !row.packed_scales ||
        !strata_iq3_derived_payload_bytes(row.blocks, &bytes) ||
        ((uintptr_t) original & 1u) || ((uintptr_t) row.d_bits & 1u)) return false;
    const block_iq3_s * source = (const block_iq3_s *) original;
    for (size_t block=0; block<row.blocks; ++block) {
        if (memcmp(row.d_bits+block, &source[block].d, 2) ||
            memcmp(row.packed_scales+block*4, source[block].scales, 4)) return false;
        for (int lane=0; lane<256; ++lane)
            if (row.levels[block*256+lane] != strata_iq3_derived_level(source+block,lane)) return false;
    }
    return true;
}

int strata_iq3_derived_compiled_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

static bool strata_iq3_derived_row_valid(int n, struct strata_iq3_derived_row row) {
    return n>0 && n%256==0 && row.blocks==(size_t) n/256 &&
        row.levels && row.d_bits && row.packed_scales && !((uintptr_t) row.d_bits & 1u);
}

#if defined(__AVX2__)
static void strata_iq3_derived_dot_core(int n, float * out0, float * out1,
        struct strata_iq3_derived_row row, const void * q8a, const void * q8b) {
    const block_q8_K * y = (const block_q8_K *) q8a;
    const block_q8_K * yb = (const block_q8_K *) q8b;
    __m256 accum0 = _mm256_setzero_ps(), accum1 = _mm256_setzero_ps();
    for (int block=0; block<n/256; ++block) {
        __m256i sum0a = _mm256_setzero_si256(), sum0b = _mm256_setzero_si256();
        __m256i sum1a = _mm256_setzero_si256(), sum1b = _mm256_setzero_si256();
        for (int group=0; group<8; group+=2) {
            const int8_t * levels = row.levels+(size_t) block*256+group*32;
            const __m256i signed0 = _mm256_loadu_si256((const __m256i *) levels);
            const __m256i signed1 = _mm256_loadu_si256((const __m256i *) (levels+32));
            const __m256i magnitude0 = _mm256_abs_epi8(signed0);
            const __m256i magnitude1 = _mm256_abs_epi8(signed1);
            const uint8_t packed = row.packed_scales[(size_t) block*4+group/2];
            const __m256i scale0 = _mm256_set1_epi16((short) (2*(packed&15)+1));
            const __m256i scale1 = _mm256_set1_epi16((short) (2*(packed>>4)+1));
            const __m256i a0 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (y[block].qs+group*32)), signed0);
            const __m256i a1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (y[block].qs+group*32+32)), signed1);
            sum0a = _mm256_add_epi32(sum0a, _mm256_madd_epi16(_mm256_maddubs_epi16(magnitude0,a0),scale0));
            sum0b = _mm256_add_epi32(sum0b, _mm256_madd_epi16(_mm256_maddubs_epi16(magnitude1,a1),scale1));
            if (yb) {
                const __m256i b0 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (yb[block].qs+group*32)), signed0);
                const __m256i b1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (yb[block].qs+group*32+32)), signed1);
                sum1a = _mm256_add_epi32(sum1a, _mm256_madd_epi16(_mm256_maddubs_epi16(magnitude0,b0),scale0));
                sum1b = _mm256_add_epi32(sum1b, _mm256_madd_epi16(_mm256_maddubs_epi16(magnitude1,b1),scale1));
            }
        }
        // Preserve canonical eight integer lanes, block FMA order and final
        // hsum. Scales are not folded into int8 levels (15*31 exceeds int8).
        const float dx = GGML_CPU_FP16_TO_FP32(row.d_bits[block]);
        const float d0 = dx*y[block].d;
        accum0 = _mm256_fmadd_ps(_mm256_set1_ps(d0),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sum0a,sum0b)),accum0);
        if (yb) {
            const float d1 = dx*yb[block].d;
            accum1 = _mm256_fmadd_ps(_mm256_set1_ps(d1),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sum1a,sum1b)),accum1);
        }
    }
    *out0 = hsum_float_8(accum0);
    if (yb) *out1 = hsum_float_8(accum1);
}
#endif

bool strata_iq3_derived_dot_single(int n, float * out,
        struct strata_iq3_derived_row row, const void * canonical_q8) {
    if (!out || !canonical_q8 || !strata_iq3_derived_row_valid(n,row)) return false;
#if defined(__AVX2__)
    strata_iq3_derived_dot_core(n,out,NULL,row,canonical_q8,NULL);
    return true;
#else
    return false;
#endif
}

bool strata_iq3_derived_dot_pair(int n, float * out0, float * out1,
        struct strata_iq3_derived_row row, const void * canonical_q8a,
        const void * canonical_q8b) {
    if (!out0 || !out1 || !canonical_q8a || !canonical_q8b ||
        !strata_iq3_derived_row_valid(n,row)) return false;
#if defined(__AVX2__)
    strata_iq3_derived_dot_core(n,out0,out1,row,canonical_q8a,canonical_q8b);
    return true;
#else
    return false;
#endif
}
