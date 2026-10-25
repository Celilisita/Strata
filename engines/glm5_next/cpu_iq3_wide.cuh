#pragma once
// Canonical IQ3_S/Q8_K block products widened to64 bytes. Fold integer halves
// before the existing8-lane FMA, preserving the original per-row float sum.
int ggml_vec_dot_iq3_s_q8_K_wide_available(void) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    return 1;
#else
    return 0;
#endif
}
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
static inline void ggml_iq3_s_wide_core(int n,float* s0,float* s1,const void* vx,const void* vy,const void* vyb) {
    const block_iq3_s* GGML_RESTRICT x=vx;
    const block_q8_K* GGML_RESTRICT y=vy;
    const block_q8_K* GGML_RESTRICT yb=vyb;
    static const uint8_t shuffle[64]={
        0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7};
    static const uint8_t bit[64]={
        1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,
        1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    static const uint32_t shift[16]={8,7,6,5,4,3,2,1,8,7,6,5,4,3,2,1};
    const __m512i shifts=_mm512_loadu_si512(shift),mask=_mm512_set1_epi32(256);
    const __m512i shuffles=_mm512_loadu_si512(shuffle),bits=_mm512_loadu_si512(bit);
    __m256 acc0=_mm256_setzero_ps(),acc1=_mm256_setzero_ps();
    for(int i=0;i<n/QK_K;++i) {
        __m512i sum0=_mm512_setzero_si512(),sum1=_mm512_setzero_si512();
        for(int ib32=0;ib32<QK_K/32;ib32+=2) {
            const __m512i low=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(x[i].qs+ib32*8)));
            const __m512i high=_mm512_mask_set1_epi32(_mm512_set1_epi32(x[i].qh[ib32]),0xff00,x[i].qh[ib32+1]);
            const __m512i index=_mm512_or_si512(low,_mm512_and_si512(_mm512_sllv_epi32(high,shifts),mask));
            const __m512i weights=_mm512_i32gather_epi32(index,iq3s_grid,4);
            uint64_t sign_bits;memcpy(&sign_bits,x[i].signs+ib32*4,sizeof(sign_bits));
            const __m512i signs=_mm512_and_si512(_mm512_shuffle_epi8(_mm512_set1_epi64((int64_t)sign_bits),shuffles),bits);
            const __m512i negative=_mm512_movm_epi8(_mm512_cmpeq_epi8_mask(signs,bits));
            const int scale0=2*(x[i].scales[ib32/2]&15)+1,scale1=2*(x[i].scales[ib32/2]>>4)+1;
            const __m512i scales=_mm512_mask_set1_epi16(_mm512_set1_epi16((short)scale0),0xffff0000u,(short)scale1);
            const __m512i a=_mm512_loadu_si512(y[i].qs+ib32*32);
            const __m512i signed_a=_mm512_sub_epi8(_mm512_xor_si512(negative,a),negative);
            sum0=_mm512_add_epi32(sum0,_mm512_madd_epi16(_mm512_maddubs_epi16(weights,signed_a),scales));
            if(yb) {
                const __m512i b=_mm512_loadu_si512(yb[i].qs+ib32*32);
                const __m512i signed_b=_mm512_sub_epi8(_mm512_xor_si512(negative,b),negative);
                sum1=_mm512_add_epi32(sum1,_mm512_madd_epi16(_mm512_maddubs_epi16(weights,signed_b),scales));
            }
        }
        const __m256i folded0=_mm256_add_epi32(_mm512_castsi512_si256(sum0),_mm512_extracti64x4_epi64(sum0,1));
        const float dx=GGML_CPU_FP16_TO_FP32(x[i].d);
        acc0=_mm256_fmadd_ps(_mm256_set1_ps(dx*y[i].d),_mm256_cvtepi32_ps(folded0),acc0);
        if(yb) {
            const __m256i folded1=_mm256_add_epi32(_mm512_castsi512_si256(sum1),_mm512_extracti64x4_epi64(sum1,1));
            acc1=_mm256_fmadd_ps(_mm256_set1_ps(dx*yb[i].d),_mm256_cvtepi32_ps(folded1),acc1);
        }
    }
    *s0=hsum_float_8(acc0);if(yb)*s1=hsum_float_8(acc1);
}
#endif
void ggml_vec_dot_iq3_s_q8_K_wide(int n,float* s,size_t bs,const void* vx,size_t bx,const void* vy,size_t by,int nrc) {
    assert(n%QK_K==0&&nrc==1);UNUSED(bs);UNUSED(bx);UNUSED(by);UNUSED(nrc);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    ggml_iq3_s_wide_core(n,s,NULL,vx,vy,NULL);
#else
    ggml_vec_dot_iq3_s_q8_K(n,s,bs,vx,bx,vy,by,nrc);
#endif
}
void ggml_vec_dot_iq3_s_q8_K_wide_pair(int n,float* s0,float* s1,const void* vx,const void* vy,const void* vyb) {
    assert(n%QK_K==0);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    ggml_iq3_s_wide_core(n,s0,s1,vx,vy,vyb);
#else
    ggml_vec_dot_iq3_s_q8_K_pair(n,s0,s1,vx,vy,vyb);
#endif
}
