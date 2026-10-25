#pragma once
// Included by the existing x86 quant TU. Original IQ4_XS payload and Q8_K
// input, same signed byte products, integer lane groups and block FMA order.
int strata_glm_cpu_iq4_packed_available(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}
#if defined(__AVX2__)
// The original four packed bytes contain low4 bits of each of eight scales.
// Multiply each copy of scales_h by a power of4, retain its low16 bits, then
// shift14: lane i is exactly (scales_h >> (2*i)) &3, including high bit15.
static inline __m128i strata_iq4_scales(const block_iq4_xs* block) {
    uint32_t packed_low;
    memcpy(&packed_low,block->scales_l,sizeof(packed_low));
    const __m128i packed=_mm_cvtsi32_si128((int32_t)packed_low);
    const __m128i mask=_mm_set1_epi8(15);
    const __m128i low=_mm_and_si128(packed,mask);
    const __m128i high=_mm_and_si128(_mm_srli_epi16(packed,4),mask);
    const __m128i nibbles=_mm_unpacklo_epi8(low,high);
    const __m128i powers=_mm_set_epi16(1,4,16,64,256,1024,4096,16384);
    const __m128i high2=_mm_srli_epi16(_mm_mullo_epi16(
        _mm_set1_epi16((short)block->scales_h),powers),14);
    return _mm_sub_epi16(_mm_or_si128(_mm_cvtepu8_epi16(nibbles),
        _mm_slli_epi16(high2,4)),_mm_set1_epi16(32));
}
#endif
void strata_glm_cpu_iq4_packed_dot(int n,float* out,size_t bs,const void* vx,size_t bx,
        const void* vy,size_t by,int nrc) {
    assert(n%QK_K==0&&nrc==1);
    UNUSED(bs);UNUSED(bx);UNUSED(by);UNUSED(nrc);
#if defined(__AVX2__)
    const block_iq4_xs* GGML_RESTRICT x=vx;
    const block_q8_K* GGML_RESTRICT y=vy;
    const __m256i values=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_iq4nl));
    const __m256i mask=_mm256_set1_epi8(15);
    __m256 accum=_mm256_setzero_ps();
    for(int block=0;block<n/QK_K;++block) {
        __m128i scales=strata_iq4_scales(x+block);
        __m256i sumi1=_mm256_setzero_si256(),sumi2=_mm256_setzero_si256();
        for(int ib=0;ib<QK_K/32;ib+=2) {
            // Each128 lane holds one packed32-weight group. Decode both
            // groups in parallel, then restore the canonical32-byte order.
            const __m256i packed=_mm256_loadu_si256((const __m256i*)(x[block].qs+ib*16));
            const __m256i low=_mm256_shuffle_epi8(values,_mm256_and_si256(packed,mask));
            const __m256i high=_mm256_shuffle_epi8(values,
                _mm256_and_si256(_mm256_srli_epi16(packed,4),mask));
            const __m256i q4a=_mm256_permute2x128_si256(low,high,0x20);
            const __m256i q4b=_mm256_permute2x128_si256(low,high,0x31);
            const __m256i q8a=_mm256_loadu_si256((const __m256i*)(y[block].qs+ib*32));
            const __m256i q8b=_mm256_loadu_si256((const __m256i*)(y[block].qs+ib*32+32));
            const __m256i p16a=mul_add_epi8(q4a,q8a);
            const __m256i p16b=mul_add_epi8(q4b,q8b);
            const __m256i sa=_mm256_broadcastw_epi16(scales);
            const __m256i sb=_mm256_broadcastw_epi16(_mm_srli_si128(scales,2));
            sumi1=_mm256_add_epi32(_mm256_madd_epi16(p16a,sa),sumi1);
            sumi2=_mm256_add_epi32(_mm256_madd_epi16(p16b,sb),sumi2);
            scales=_mm_srli_si128(scales,4);
        }
        accum=_mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[block].d)*y[block].d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1,sumi2)),accum);
    }
    *out=hsum_float_8(accum);
#else
    ggml_vec_dot_iq4_xs_q8_K(n,out,bs,vx,bx,vy,by,nrc);
#endif
}
