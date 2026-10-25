// FixedNy instantiations. LUTs build once per256block/activation, inside this
// measured128row leaf; each table is shared across allfour32row tiles.
static void STRATA_IQ3_LUT_KERNEL(float * out0, float * out1, const void * original,
        size_t row_stride, const block_q8_K * q8a, const block_q8_K * q8b) {
    uint8_t low[STRATA_IQ3_LUT_NY][64][16], high[STRATA_IQ3_LUT_NY][64][16];
    float accumulator[STRATA_IQ3_LUT_NY][128] = {{0}};
    const block_q8_K * activation[2] = {q8a,q8b};
    for (int block = 0; block < 16; ++block) {
        for (int iy = 0; iy < STRATA_IQ3_LUT_NY; ++iy)
            strata_iq3_lut_build(activation[iy][block].qs,low[iy],high[iy]);
        for (int tile = 0; tile < 128; tile += 32) {
            __m256i total[STRATA_IQ3_LUT_NY][4];
            float d[32];
            const block_iq3_s * source[32];
            for (int row = 0; row < 32; ++row) {
                source[row] = (const block_iq3_s *)((const char *)original+(tile+row)*row_stride)+block;
                d[row] = GGML_CPU_FP16_TO_FP32(source[row]->d);
            }
            for (int iy = 0; iy < STRATA_IQ3_LUT_NY; ++iy)
                for (int v = 0; v < 4; ++v) total[iy][v] = _mm256_setzero_si256();
            for (int group = 0; group < 8; ++group) {
                __m256i group_sum[STRATA_IQ3_LUT_NY][4], half[STRATA_IQ3_LUT_NY][2];
                int32_t scales[32];
                for (int row = 0; row < 32; ++row)
                    scales[row] = 2*((source[row]->scales[group/2]>>(4*(group%2)))&15)+1;
                for (int iy = 0; iy < STRATA_IQ3_LUT_NY; ++iy) {
                    for (int v = 0; v < 4; ++v) group_sum[iy][v] = _mm256_setzero_si256();
                    half[iy][0] = half[iy][1] = _mm256_setzero_si256();
                }
                for (int quartet = 0; quartet < 8; ++quartet) {
                    uint8_t indices[4][32];
                    for (int row = 0; row < 32; ++row) {
                        const block_iq3_s * weight = source[row];
                        const unsigned index = weight->qs[8*group+quartet] |
                            (((unsigned)weight->qh[group]>>(unsigned)quartet)&1u)<<8;
                        const unsigned signs = (weight->signs[4*group+quartet/2]>>(4*(quartet%2)))&15;
                        const unsigned planes = strata_iq3_lut_planes[index]^(signs*0x1111u);
                        for (int p = 0; p < 4; ++p) indices[p][row] = (uint8_t)((planes>>(4*p))&15);
                    }
                    for (int iy = 0; iy < STRATA_IQ3_LUT_NY; ++iy) {
                        const __m256i lo = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)low[iy][8*group+quartet]),
                                                        _mm_loadu_si128((const __m128i *)low[iy][8*group+quartet]));
                        const __m256i hi = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)high[iy][8*group+quartet]),
                                                        _mm_loadu_si128((const __m128i *)high[iy][8*group+quartet]));
                        struct strata_iq3_lut_lookup value = strata_iq3_lut_lookup16(lo,hi,indices[0]);
                        struct strata_iq3_lut_lookup plane = strata_iq3_lut_lookup16(lo,hi,indices[1]);
                        value.low_rows = _mm256_add_epi16(value.low_rows,_mm256_slli_epi16(plane.low_rows,1));
                        value.high_rows = _mm256_add_epi16(value.high_rows,_mm256_slli_epi16(plane.high_rows,1));
                        plane = strata_iq3_lut_lookup16(lo,hi,indices[2]);
                        value.low_rows = _mm256_add_epi16(value.low_rows,_mm256_slli_epi16(plane.low_rows,2));
                        value.high_rows = _mm256_add_epi16(value.high_rows,_mm256_slli_epi16(plane.high_rows,2));
                        plane = strata_iq3_lut_lookup16(lo,hi,indices[3]);
                        value.low_rows = _mm256_add_epi16(value.low_rows,_mm256_slli_epi16(plane.low_rows,3));
                        value.high_rows = _mm256_add_epi16(value.high_rows,_mm256_slli_epi16(plane.high_rows,3));
                        half[iy][0] = _mm256_add_epi16(half[iy][0],value.low_rows);
                        half[iy][1] = _mm256_add_epi16(half[iy][1],value.high_rows);
                        // Atmostfour quartets:30480fitsint16. Widen before32values/scale.
                        if (quartet == 3 || quartet == 7) {
                            group_sum[iy][0] = _mm256_add_epi32(group_sum[iy][0],_mm256_cvtepi16_epi32(_mm256_castsi256_si128(half[iy][0])));
                            group_sum[iy][1] = _mm256_add_epi32(group_sum[iy][1],_mm256_cvtepi16_epi32(_mm256_castsi256_si128(half[iy][1])));
                            group_sum[iy][2] = _mm256_add_epi32(group_sum[iy][2],_mm256_cvtepi16_epi32(_mm256_extractf128_si256(half[iy][0],1)));
                            group_sum[iy][3] = _mm256_add_epi32(group_sum[iy][3],_mm256_cvtepi16_epi32(_mm256_extractf128_si256(half[iy][1],1)));
                            half[iy][0] = half[iy][1] = _mm256_setzero_si256();
                        }
                    }
                }
                for (int iy = 0; iy < STRATA_IQ3_LUT_NY; ++iy) for (int v = 0; v < 4; ++v)
                    total[iy][v] = _mm256_add_epi32(total[iy][v],_mm256_mullo_epi32(group_sum[iy][v],
                        _mm256_loadu_si256((const __m256i *)(scales+8*v))));
            }
            for (int iy = 0; iy < STRATA_IQ3_LUT_NY; ++iy) for (int v = 0; v < 4; ++v) {
                const __m256 factor = _mm256_mul_ps(_mm256_loadu_ps(d+8*v),_mm256_set1_ps(activation[iy][block].d));
                const __m256 old = _mm256_loadu_ps(accumulator[iy]+tile+8*v);
                _mm256_storeu_ps(accumulator[iy]+tile+8*v,_mm256_fmadd_ps(factor,_mm256_cvtepi32_ps(total[iy][v]),old));
            }
        }
    }
    memcpy(out0,accumulator[0],128*sizeof(float));
#if STRATA_IQ3_LUT_NY == 2
    memcpy(out1,accumulator[1],128*sizeof(float));
#else
    GGML_UNUSED(out1);
#endif
}
