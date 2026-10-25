// Intentionally included twice:Ny1/Ny2 have fixed independent accumulators.
// Row-lane mapping follows pinned ik_llama.cpp IQ3_S_R4; see MIT notice.
STRATA_IQ3_R4_TARGET static void STRATA_IQ3_R4_KERNEL(int n, float output[STRATA_IQ3_R4_NY][4],
        const struct strata_iq3_r4_block * x, const block_q8_K * const y[2]) {
    const __m256i one = _mm256_set1_epi8(1);
    __m256 accumulator[STRATA_IQ3_R4_NY];
    for (int iy = 0; iy < STRATA_IQ3_R4_NY; ++iy) accumulator[iy] = _mm256_setzero_ps();
    for (int block = 0; block < n/256; ++block) {
        float d[4];
        for (int k = 0; k < 4; ++k) d[k] = GGML_CPU_FP16_TO_FP32(x[block].d[k]);
        const __m128 d4 = _mm_loadu_ps(d);
        const __m256 scale_d = _mm256_castsi256_ps(MM256_SET_M128I(_mm_castps_si128(d4),_mm_castps_si128(d4)));
        const __m128i packed_scale = _mm_loadu_si128((const __m128i *)x[block].scales);
        const __m256i scale8 = MM256_SET_M128I(_mm_srli_epi16(packed_scale,4),packed_scale);
        union { __m256i v; uint32_t a[8]; } scales;
        scales.v = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(scale8,
            _mm256_set1_epi8(15)),1),one);
        __m256i integer_sum[STRATA_IQ3_R4_NY];
        for (int iy = 0; iy < STRATA_IQ3_R4_NY; ++iy) integer_sum[iy] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t qh;
            memcpy(&qh,x[block].qh+4*ib,sizeof(qh));
            __m128i high = _mm_sllv_epi64(_mm_cvtepu8_epi16(_mm_set1_epi32((int)qh)),
                                        _mm_set_epi64x(4,8));
            __m256i weights[4], signs[4];
            for (int i = 0; i < 4; ++i) {
                const __m128i low = _mm_cvtepu8_epi16(_mm_loadl_epi64(
                    (const __m128i *)(x[block].qs+32*ib+8*i)));
                const __m128i index = _mm_or_si128(low,_mm_and_si128(high,_mm_set1_epi16(256)));
                high = _mm_srli_epi16(high,1);
                weights[i] = _mm256_i32gather_epi32((const int *)iq3s_grid,_mm256_cvtepu16_epi32(index),4);
            }
            const __m128i sign128 = _mm_loadu_si128((const __m128i *)(x[block].signs+16*ib));
            __m256i sign_bits = MM256_SET_M128I(_mm_srli_epi16(sign128,4),sign128);
            for (int i = 0; i < 4; ++i) {
                signs[i] = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sign_bits,one),one),one);
                sign_bits = _mm256_srli_epi16(sign_bits,1);
            }
            const __m128i four_scales = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)scales.a[ib]));
            const __m256i scale = MM256_SET_M128I(four_scales,four_scales);
            for (int iy = 0; iy < STRATA_IQ3_R4_NY; ++iy) {
                const __m256i activation = _mm256_loadu_si256((const __m256i *)(y[iy][block].qs+32*ib));
                __m256i sum = _mm256_setzero_si256();
                sum = _mm256_dpbusd_epi32(sum,weights[0],_mm256_sign_epi8(_mm256_shuffle_epi32(activation,0x00),signs[0]));
                sum = _mm256_dpbusd_epi32(sum,weights[1],_mm256_sign_epi8(_mm256_shuffle_epi32(activation,0x55),signs[1]));
                sum = _mm256_dpbusd_epi32(sum,weights[2],_mm256_sign_epi8(_mm256_shuffle_epi32(activation,0xaa),signs[2]));
                sum = _mm256_dpbusd_epi32(sum,weights[3],_mm256_sign_epi8(_mm256_shuffle_epi32(activation,0xff),signs[3]));
                integer_sum[iy] = _mm256_add_epi32(integer_sum[iy],_mm256_mullo_epi32(sum,scale));
            }
        }
        for (int iy = 0; iy < STRATA_IQ3_R4_NY; ++iy) {
            const __m256 factor = _mm256_mul_ps(scale_d,_mm256_set1_ps(y[iy][block].d));
            accumulator[iy] = _mm256_fmadd_ps(factor,_mm256_cvtepi32_ps(integer_sum[iy]),accumulator[iy]);
        }
    }
    for (int iy = 0; iy < STRATA_IQ3_R4_NY; ++iy) {
        const __m128 result = _mm_add_ps(_mm256_castps256_ps128(accumulator[iy]),
                                       _mm256_extractf128_ps(accumulator[iy],1));
        _mm_storeu_ps(output[iy],result);
    }
}
