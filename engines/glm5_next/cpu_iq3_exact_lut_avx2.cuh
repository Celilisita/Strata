#pragma once
#include "cpu_iq3_exact_lut.h"
#include "cpu_iq3_exact_lut_grid.h"
#if defined(__AVX2__)
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

bool strata_iq3_exact_lut_grid_identity(void) {
    for (size_t index = 0; index < 512; ++index) {
        uint16_t expected = 0;
        for (unsigned lane = 0; lane < 4; ++lane) {
            const unsigned value = (iq3s_grid[index]>>(8*lane))&255;
            if (!(value&1) || value > 15) return false;
            const unsigned code = (value+15)/2;
            for (unsigned bit = 0; bit < 4; ++bit) expected |= ((code>>bit)&1)<<(4*bit+lane);
        }
        if (expected != strata_iq3_lut_planes[index]) return false;
    }
    return true;
}

#if defined(__AVX2__)
struct strata_iq3_lut_lookup {
    __m256i low_rows, high_rows; // rows0..7/16..23 and8..15/24..31.
};
static inline struct strata_iq3_lut_lookup strata_iq3_lut_lookup16(
        __m256i low, __m256i high, const uint8_t * indices) {
    const __m256i index = _mm256_loadu_si256((const __m256i *)indices);
    const __m256i lo = _mm256_shuffle_epi8(low,index), hi = _mm256_shuffle_epi8(high,index);
    struct strata_iq3_lut_lookup result = {_mm256_unpacklo_epi8(lo,hi),_mm256_unpackhi_epi8(lo,hi)};
    return result;
}
// Values are exact +/- sums of four Q8 integers, never requantized.
static void strata_iq3_lut_build(const int8_t * q8, uint8_t low[64][16], uint8_t high[64][16]) {
    for (int quartet = 0; quartet < 64; ++quartet) {
        const int8_t * x = q8+4*quartet;
        int16_t value[8];
        value[0] = (int16_t)(-x[0]-x[1]-x[2]-x[3]);
        value[1] = (int16_t)(value[0]+2*x[0]);
        value[2] = (int16_t)(value[0]+2*x[1]);
        value[3] = (int16_t)(value[1]+2*x[1]);
        value[4] = (int16_t)(value[0]+2*x[2]);
        value[5] = (int16_t)(value[1]+2*x[2]);
        value[6] = (int16_t)(value[2]+2*x[2]);
        value[7] = (int16_t)(value[3]+2*x[2]);
        for (int code = 0; code < 8; ++code) {
            const uint16_t a = (uint16_t)value[code], b = (uint16_t)-value[code];
            low[quartet][code] = (uint8_t)a; high[quartet][code] = (uint8_t)(a>>8);
            low[quartet][15-code] = (uint8_t)b; high[quartet][15-code] = (uint8_t)(b>>8);
        }
    }
}
#define STRATA_IQ3_LUT_NY 1
#define STRATA_IQ3_LUT_KERNEL strata_iq3_exact_lut_ny1
#include "cpu_iq3_exact_lut_kernel.inc"
#undef STRATA_IQ3_LUT_KERNEL
#undef STRATA_IQ3_LUT_NY
#define STRATA_IQ3_LUT_NY 2
#define STRATA_IQ3_LUT_KERNEL strata_iq3_exact_lut_ny2
#include "cpu_iq3_exact_lut_kernel.inc"
#undef STRATA_IQ3_LUT_KERNEL
#undef STRATA_IQ3_LUT_NY
#endif

unsigned strata_iq3_exact_lut_capabilities(void) {
#if defined(__AVX2__)
    unsigned leaf1c, leaf7b, maxleaf; uint64_t xcr0 = 0;
#if defined(_MSC_VER) && !defined(__clang__)
    int regs[4]; __cpuid(regs,0); maxleaf = (unsigned)regs[0];
    if (maxleaf < 7) return 1;
    __cpuidex(regs,1,0); leaf1c = (unsigned)regs[2];
    if ((leaf1c&0x1c000000u) == 0x1c000000u) xcr0 = _xgetbv(0);
    __cpuidex(regs,7,0); leaf7b = (unsigned)regs[1];
#else
    unsigned a,b,c,d; maxleaf = __get_cpuid_max(0,NULL);
    if (maxleaf < 7) return 1;
    __cpuid_count(1,0,a,b,c,d); leaf1c = c;
    if ((c&0x1c000000u) == 0x1c000000u) {
        unsigned lo,hi; __asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0)); xcr0 = ((uint64_t)hi<<32)|lo;
    }
    __cpuid_count(7,0,a,b,c,d); leaf7b = b;
#endif
    const unsigned required = (1u<<12)|(1u<<28)|(1u<<29);
    const bool cpu = (leaf1c&required) == required && (leaf7b&(1u<<5));
    const bool os = (leaf1c&0x1c000000u) == 0x1c000000u && (xcr0&6u) == 6u;
    return 1u|(cpu?2u:0u)|(os?4u:0u);
#else
    return 0;
#endif
}
bool strata_iq3_exact_lut_chunk(int n, float * out0, float * out1, const void * original,
        size_t row_stride, const void * q8a, const void * q8b, size_t rows) {
    if (n != 4096 || rows != 128 || row_stride != 1760 || !out0 || !original || !q8a ||
        ((out1 == NULL) != (q8b == NULL))) return false;
#if defined(__AVX2__)
    if (out1) strata_iq3_exact_lut_ny2(out0,out1,original,row_stride,q8a,q8b);
    else strata_iq3_exact_lut_ny1(out0,NULL,original,row_stride,q8a,NULL);
    return true;
#else
    return false;
#endif
}
