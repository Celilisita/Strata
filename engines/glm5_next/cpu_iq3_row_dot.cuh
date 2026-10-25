#pragma once
// Canonical IQ3_S/Q8_K arithmetic over two output rows. Activations are loaded
// once per64 elements; each output retains its independent eight-lane block FMA.
#include "cpu_row_dot.h"
int strata_glm_cpu_row_dot_available(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}
#if defined(__AVX2__)
#ifdef _MSC_VER
#define STRATA_ROW_INLINE static __forceinline
#else
#define STRATA_ROW_INLINE static inline __attribute__((always_inline))
#endif
struct strata_row_product { __m256i a,b; };
STRATA_ROW_INLINE struct strata_row_product strata_row_product64(const block_iq3_s* x,int ib,
        __m256i a0,__m256i a1,__m256i b0,__m256i b1,bool dual,
        __m256i mask1,__m256i mask2,__m256i shift,__m256i mask) {
    union { __m256i v[2];uint32_t i[16]; } index;
    const __m256i low=_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(x->qs+ib*8)));
    index.v[0]=_mm256_or_si256(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(low)),
        _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(x->qh[ib]),shift),mask));
    index.v[1]=_mm256_or_si256(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(low,1)),
        _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(x->qh[ib+1]),shift),mask));
    const __m256i w0=_mm256_set_epi32(iq3s_grid[index.i[7]],iq3s_grid[index.i[6]],iq3s_grid[index.i[5]],iq3s_grid[index.i[4]],
        iq3s_grid[index.i[3]],iq3s_grid[index.i[2]],iq3s_grid[index.i[1]],iq3s_grid[index.i[0]]);
    const __m256i w1=_mm256_set_epi32(iq3s_grid[index.i[15]],iq3s_grid[index.i[14]],iq3s_grid[index.i[13]],iq3s_grid[index.i[12]],
        iq3s_grid[index.i[11]],iq3s_grid[index.i[10]],iq3s_grid[index.i[9]],iq3s_grid[index.i[8]]);
    const uint16_t* signs=(const uint16_t*)(x->signs+ib*4);
    const __m256i s0=_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(
        _mm256_set1_epi32((int32_t)((uint32_t)signs[0]|((uint32_t)signs[1]<<16))),mask1),mask2),mask2);
    const __m256i s1=_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(
        _mm256_set1_epi32((int32_t)((uint32_t)signs[2]|((uint32_t)signs[3]<<16))),mask1),mask2),mask2);
    const __m256i scale0=_mm256_set1_epi16(2*(x->scales[ib/2]&15)+1);
    const __m256i scale1=_mm256_set1_epi16(2*(x->scales[ib/2]>>4)+1);
    struct strata_row_product result;
    result.a=_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_maddubs_epi16(w0,_mm256_sub_epi8(_mm256_xor_si256(s0,a0),s0)),scale0),
        _mm256_madd_epi16(_mm256_maddubs_epi16(w1,_mm256_sub_epi8(_mm256_xor_si256(s1,a1),s1)),scale1));
    result.b=_mm256_setzero_si256();
    if(dual)result.b=_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_maddubs_epi16(w0,_mm256_sub_epi8(_mm256_xor_si256(s0,b0),s0)),scale0),
        _mm256_madd_epi16(_mm256_maddubs_epi16(w1,_mm256_sub_epi8(_mm256_xor_si256(s1,b1),s1)),scale1));
    return result;
}
static void strata_row_dot2(int n,float* out0,float* out1,const block_iq3_s* x0,const block_iq3_s* x1,
        const block_q8_K* y,const block_q8_K* yb) {
    static const uint8_t m1[32]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3};
    static const uint8_t m2[32]={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,
        1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    const __m256i mask1=_mm256_loadu_si256((const __m256i*)m1),mask2=_mm256_loadu_si256((const __m256i*)m2);
    const __m256i shift=_mm256_set_epi32(1,2,3,4,5,6,7,8),mask=_mm256_set1_epi32(256);
    __m256 acc00=_mm256_setzero_ps(),acc10=_mm256_setzero_ps();
    __m256 acc01=_mm256_setzero_ps(),acc11=_mm256_setzero_ps();
    for(int block=0;block<n/QK_K;++block) {
        const float dx0=GGML_CPU_FP16_TO_FP32(x0[block].d),dx1=GGML_CPU_FP16_TO_FP32(x1[block].d);
        const float d00=dx0*y[block].d,d10=dx1*y[block].d;
        const float d01=yb?dx0*yb[block].d:0,d11=yb?dx1*yb[block].d:0;
        __m256i sum00=_mm256_setzero_si256(),sum10=_mm256_setzero_si256();
        __m256i sum01=_mm256_setzero_si256(),sum11=_mm256_setzero_si256();
        for(int ib=0;ib<QK_K/32;ib+=2) {
            const __m256i a0=_mm256_loadu_si256((const __m256i*)(y[block].qs+ib*32));
            const __m256i a1=_mm256_loadu_si256((const __m256i*)(y[block].qs+ib*32+32));
            const __m256i b0=yb?_mm256_loadu_si256((const __m256i*)(yb[block].qs+ib*32)):_mm256_setzero_si256();
            const __m256i b1=yb?_mm256_loadu_si256((const __m256i*)(yb[block].qs+ib*32+32)):_mm256_setzero_si256();
            const struct strata_row_product p0=strata_row_product64(x0+block,ib,a0,a1,b0,b1,yb!=NULL,mask1,mask2,shift,mask);
            const struct strata_row_product p1=strata_row_product64(x1+block,ib,a0,a1,b0,b1,yb!=NULL,mask1,mask2,shift,mask);
            // Int32 lane addition is exact modulo2^32: folding the two integer
            // substreams here preserves the original block sum before its FMA.
            sum00=_mm256_add_epi32(sum00,p0.a);sum10=_mm256_add_epi32(sum10,p1.a);
            if(yb){sum01=_mm256_add_epi32(sum01,p0.b);sum11=_mm256_add_epi32(sum11,p1.b);}
        }
        acc00=_mm256_fmadd_ps(_mm256_set1_ps(d00),_mm256_cvtepi32_ps(sum00),acc00);
        acc10=_mm256_fmadd_ps(_mm256_set1_ps(d10),_mm256_cvtepi32_ps(sum10),acc10);
        if(yb) {
            acc01=_mm256_fmadd_ps(_mm256_set1_ps(d01),_mm256_cvtepi32_ps(sum01),acc01);
            acc11=_mm256_fmadd_ps(_mm256_set1_ps(d11),_mm256_cvtepi32_ps(sum11),acc11);
        }
    }
    out0[0]=hsum_float_8(acc00);out0[1]=hsum_float_8(acc10);
    if(yb){out1[0]=hsum_float_8(acc01);out1[1]=hsum_float_8(acc11);}
}
#undef STRATA_ROW_INLINE
#endif
uint64_t strata_glm_cpu_row_dot(int n,float* out0,float* out1,const void* rows,size_t stride,
        const void* activation0,const void* activation1,int count) {
    assert(n%QK_K==0&&count>0&&out0&&rows&&activation0&&(!activation1||out1));
    int row=0;uint64_t pairs=0,fallback=0;
#if defined(__AVX2__)
    for(;row+1<count;row+=2) {
        strata_row_dot2(n,out0+row,out1?out1+row:NULL,
            (const block_iq3_s*)((const char*)rows+row*stride),
            (const block_iq3_s*)((const char*)rows+(row+1)*stride),activation0,activation1);
        ++pairs;
    }
#endif
    for(;row<count;++row) {
        const void* x=(const char*)rows+row*stride;
        if(activation1)ggml_vec_dot_iq3_s_q8_K_pair(n,out0+row,out1+row,x,activation0,activation1);
        else ggml_vec_dot_iq3_s_q8_K(n,out0+row,0,x,0,activation0,0,1);
        ++fallback;
    }
    return pairs|(fallback<<32);
}
