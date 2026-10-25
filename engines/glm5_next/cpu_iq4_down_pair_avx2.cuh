#pragma once
#include "cpu_iq4_down_pair.h"
// Included by the original x86 quants TU. This copies its IQ4_XS arithmetic
// order per activation, sharing only original weight decoding between them.
int strata_glm_cpu_iq4_down_pair_available(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}
void strata_glm_cpu_iq4_down_pair(int n, float * out0, float * out1,
        const void * original, const void * q8a, const void * q8b) {
    assert(n % QK_K == 0);
#if defined(__AVX2__)
    const block_iq4_xs * x = (const block_iq4_xs *) original;
    const block_q8_K * y0 = (const block_q8_K *) q8a;
    const block_q8_K * y1 = (const block_q8_K *) q8b;
    const __m128i values = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i mask = _mm_set1_epi8(15);
    __m256 accum0 = _mm256_setzero_ps(), accum1 = _mm256_setzero_ps();
    for (int block = 0; block < n/QK_K; ++block) {
        const uint8_t * qs = x[block].qs;
        const int8_t * qa = y0[block].qs, * qb = y1[block].qs;
        uint16_t sh = x[block].scales_h;
        __m256i sum01 = _mm256_setzero_si256(), sum02 = _mm256_setzero_si256();
        __m256i sum11 = _mm256_setzero_si256(), sum12 = _mm256_setzero_si256();
        for (int group = 0; group < QK_K/32; group += 2) {
            const __m128i bits1 = _mm_loadu_si128((const __m128i *) qs); qs += 16;
            const __m128i bits2 = _mm_loadu_si128((const __m128i *) qs); qs += 16;
            const __m256i weights1 = MM256_SET_M128I(
                _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits1,4),mask)),
                _mm_shuffle_epi8(values, _mm_and_si128(bits1,mask)));
            const __m256i weights2 = MM256_SET_M128I(
                _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits2,4),mask)),
                _mm_shuffle_epi8(values, _mm_and_si128(bits2,mask)));
            const int16_t ls1 = ((x[block].scales_l[group/2]&15)|((sh<<4)&48))-32;
            const int16_t ls2 = ((x[block].scales_l[group/2]>>4)|((sh<<2)&48))-32;
            sh >>= 4;
            const __m256i scale1 = _mm256_set1_epi16(ls1), scale2 = _mm256_set1_epi16(ls2);
            const __m256i a1 = _mm256_loadu_si256((const __m256i *) qa); qa += 32;
            const __m256i a2 = _mm256_loadu_si256((const __m256i *) qa); qa += 32;
            const __m256i b1 = _mm256_loadu_si256((const __m256i *) qb); qb += 32;
            const __m256i b2 = _mm256_loadu_si256((const __m256i *) qb); qb += 32;
            sum01 = _mm256_add_epi32(_mm256_madd_epi16(mul_add_epi8(weights1,a1),scale1),sum01);
            sum02 = _mm256_add_epi32(_mm256_madd_epi16(mul_add_epi8(weights2,a2),scale2),sum02);
            sum11 = _mm256_add_epi32(_mm256_madd_epi16(mul_add_epi8(weights1,b1),scale1),sum11);
            sum12 = _mm256_add_epi32(_mm256_madd_epi16(mul_add_epi8(weights2,b2),scale2),sum12);
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[block].d);
        accum0 = _mm256_fmadd_ps(_mm256_set1_ps(d*y0[block].d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sum01,sum02)),accum0);
        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(d*y1[block].d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sum11,sum12)),accum1);
    }
    *out0 = hsum_float_8(accum0);
    *out1 = hsum_float_8(accum1);
#else
    ggml_vec_dot_iq4_xs_q8_K(n,out0,0,original,0,q8a,0,1);
    ggml_vec_dot_iq4_xs_q8_K(n,out1,0,original,0,q8b,0,1);
#endif
}
