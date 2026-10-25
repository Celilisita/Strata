#pragma once
#include "cpu_iq3_r4.h"
#include "cpu_iq3_r4_license.h"

// Independent compile macros:the earlier VNNI header intentionally undefines its own.
// Conditions match that existing backend; runtime capability remains its CPUID/OS check.
#if defined(__AVX2__) && defined(_MSC_VER) && _MSC_VER >= 1920 && defined(_M_X64) && !defined(__clang__) && !defined(_M_ARM64EC)
#define STRATA_IQ3_R4_COMPILED 1
#define STRATA_IQ3_R4_TARGET
#elif defined(__AVX2__) && !defined(_MSC_VER) && (defined(__x86_64__) || defined(__i386__)) && ((defined(__clang__) && __clang_major__ >= 7) || (defined(__GNUC__) && __GNUC__ >= 8))
#define STRATA_IQ3_R4_COMPILED 1
#define STRATA_IQ3_R4_TARGET __attribute__((target("avx512vnni,avx512vl,avx512bw,avx512dq,avx2,fma,f16c")))
#else
#define STRATA_IQ3_R4_COMPILED 0
#define STRATA_IQ3_R4_TARGET
#endif
#if STRATA_IQ3_R4_COMPILED
#define STRATA_IQ3_R4_NY 1
#define STRATA_IQ3_R4_KERNEL strata_iq3_r4_ny1
#include "cpu_iq3_r4_kernel.inc"
#undef STRATA_IQ3_R4_KERNEL
#undef STRATA_IQ3_R4_NY
#define STRATA_IQ3_R4_NY 2
#define STRATA_IQ3_R4_KERNEL strata_iq3_r4_ny2
#include "cpu_iq3_r4_kernel.inc"
#undef STRATA_IQ3_R4_KERNEL
#undef STRATA_IQ3_R4_NY
#endif

unsigned strata_iq3_r4_capabilities(void) {
#if STRATA_IQ3_R4_COMPILED
    return strata_glm_cpu_vnni_capabilities();
#else
    return 0;
#endif
}
bool strata_iq3_r4_dot(int n, float * out0, size_t stride0, float * out1, size_t stride1,
        struct strata_iq3_r4_rows rows, const void * q8a, const void * q8b) {
    if (n != 4096 || !rows.data || rows.blocks != 16 || !out0 || !q8a ||
        stride0 < sizeof(float) || stride0%sizeof(float) || stride0 > SIZE_MAX/4 ||
        ((out1 == NULL) != (q8b == NULL)) ||
        (out1 && (stride1 < sizeof(float) || stride1%sizeof(float) || stride1 > SIZE_MAX/4))) return false;
#if STRATA_IQ3_R4_COMPILED
    float values[2][4];
    const block_q8_K * activation[2] = {q8a,q8b};
    if (out1) strata_iq3_r4_ny2(n,values,rows.data,activation);
    else strata_iq3_r4_ny1(n,values,rows.data,activation);
    for (size_t k = 0; k < 4; ++k) {
        memcpy((char *)out0+k*stride0,&values[0][k],sizeof(float));
        if (out1) memcpy((char *)out1+k*stride1,&values[1][k],sizeof(float));
    }
    return true;
#else
    return false;
#endif
}

#undef STRATA_IQ3_R4_TARGET
#undef STRATA_IQ3_R4_COMPILED
