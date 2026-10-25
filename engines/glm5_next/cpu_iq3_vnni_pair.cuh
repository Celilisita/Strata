#pragma once
// IQ3_S single/pair opt-in. Non-saturating int16 dot+int32 add fusion; original
// maddubs, grid/sign/scales and each output's block FMA sequence are retained.
#include "cpu_vnni_dot.h"
#if defined(__AVX2__) && defined(_MSC_VER) && _MSC_VER >= 1920 && defined(_M_X64) && !defined(__clang__) && !defined(_M_ARM64EC)
#include <intrin.h>
#define STRATA_VNNI_COMPILED 1
#define STRATA_VNNI_TARGET
#elif defined(__AVX2__) && !defined(_MSC_VER) && (defined(__x86_64__) || defined(__i386__)) && ((defined(__clang__) && __clang_major__ >= 7) || (defined(__GNUC__) && __GNUC__ >= 8))
#include <cpuid.h>
#define STRATA_VNNI_COMPILED 1
#define STRATA_VNNI_TARGET __attribute__((target("avx512vnni,avx512vl,avx512bw,avx512dq,avx2,fma,f16c")))
#else
#define STRATA_VNNI_COMPILED 0
#define STRATA_VNNI_TARGET
#endif
unsigned strata_glm_cpu_vnni_capabilities(void) {
#if STRATA_VNNI_COMPILED
    unsigned a,b,c,d,leaf1c;uint64_t xcr0=0;
#if defined(_MSC_VER) && !defined(__clang__)
    int regs[4];__cpuid(regs,0);if(regs[0]<7)return 1;
    __cpuidex(regs,1,0);c=(unsigned)regs[2];
    leaf1c=c;
    if((c&0x1c000000u)==0x1c000000u)xcr0=_xgetbv(0);
    __cpuidex(regs,7,0);b=(unsigned)regs[1];c=(unsigned)regs[2];
#else
    if(__get_cpuid_max(0,NULL)<7)return 1;
    __cpuid_count(1,0,a,b,c,d);
    leaf1c=c;
    if((c&0x1c000000u)==0x1c000000u) {
        unsigned lo,hi;__asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));xcr0=((uint64_t)hi<<32)|lo;
    }
    __cpuid_count(7,0,a,b,c,d);
#endif
    return strata_glm_cpu_vnni_decode(leaf1c,b,c,xcr0);
#else
    return 0;
#endif
}
STRATA_VNNI_TARGET void strata_glm_cpu_iq3_s_vnni_pair(int n,float* s0,float* s1,const void* vx,const void* vy,const void* vyb) {
    assert(n%QK_K==0);
#if STRATA_VNNI_COMPILED
    const block_iq3_s* GGML_RESTRICT x=vx;
    const block_q8_K* GGML_RESTRICT y=vy;
    const block_q8_K* GGML_RESTRICT yb=vyb;
    const int nb=n/QK_K;
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = _mm256_setzero_ps(),accumf_b = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float d=dx*y[i].d,db=dx*yb[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        __m256i sumi1_b=_mm256_setzero_si256(),sumi2_b=_mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, _mm256_loadu_si256((const __m256i*)(y[i].qs+ib32*32))), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, _mm256_loadu_si256((const __m256i*)(y[i].qs+ib32*32+32))), s2_2);

            const __m256i q8b_1=_mm256_loadu_si256((const __m256i*)(yb[i].qs+ib32*32));
            const __m256i q8b_2=_mm256_loadu_si256((const __m256i*)(yb[i].qs+ib32*32+32));
            const __m256i q8sb_1=_mm256_sub_epi8(_mm256_xor_si256(s2_1,q8b_1),s2_1);
            const __m256i q8sb_2=_mm256_sub_epi8(_mm256_xor_si256(s2_2,q8b_2),s2_2);
            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            sumi1=_mm256_dpwssd_epi32(sumi1,dot1,_mm256_set1_epi16(2*ls1+1));
            sumi2=_mm256_dpwssd_epi32(sumi2,dot2,_mm256_set1_epi16(2*ls2+1));
            sumi1_b=_mm256_dpwssd_epi32(sumi1_b,_mm256_maddubs_epi16(q2_1,q8sb_1),_mm256_set1_epi16(2*ls1+1));
            sumi2_b=_mm256_dpwssd_epi32(sumi2_b,_mm256_maddubs_epi16(q2_2,q8sb_2),_mm256_set1_epi16(2*ls2+1));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
        accumf_b=_mm256_fmadd_ps(_mm256_set1_ps(db),_mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_b,sumi2_b)),accumf_b);

    }

    *s0=hsum_float_8(accumf);*s1=hsum_float_8(accumf_b);

#else
    ggml_vec_dot_iq3_s_q8_K_pair(n,s0,s1,vx,vy,vyb);
#endif
}
STRATA_VNNI_TARGET void strata_glm_cpu_iq3_s_vnni(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if STRATA_VNNI_COMPILED

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            sumi1 = _mm256_dpwssd_epi32(sumi1, dot1, _mm256_set1_epi16(2*ls1+1));
            sumi2 = _mm256_dpwssd_epi32(sumi2, dot2, _mm256_set1_epi16(2*ls2+1));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = hsum_float_8(accumf);

#else
    UNUSED(x);UNUSED(y);UNUSED(nb);
    ggml_vec_dot_iq3_s_q8_K(n,s,bs,vx,bx,vy,by,nrc);
#endif
}
#undef STRATA_VNNI_TARGET
#undef STRATA_VNNI_COMPILED
