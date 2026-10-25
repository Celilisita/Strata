#pragma once
#include "ggml-backend.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Private chunk ABI, never cast to ggml_vec_dot_t. No allocation or pool state.
// table[64 quartets][16 signs][512 grids]: low/high signed16 are Q8 classes0/1.
// Caller publishes the complete table and retires all readers before reuse.
enum { STRATA_IQ3_DUAL_LUT_ROWS=128, STRATA_IQ3_DUAL_LUT_WORDS=524288,
       STRATA_IQ3_DUAL_LUT_BYTES=2097152 };
#ifdef __cplusplus
extern "C" {
#endif
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
// Setup only: bit0 compiledAVX2; bit1 CPU AVX2/FMA/F16C; bit2 OS XMM/YMM.
GGML_BACKEND_API unsigned strata_iq3_dual_lut_capabilities(void);
// Once per actual canonical class/job, before committing to the barrier loop.
// Checks all16 d/qs blocks; ignores bsums, which this original IQ3 dot never reads.
// Reject nonfinite d and -128: original byte-sign negation wraps -128, unlike
// mathematical signed integer negation. Canonical finite Q8_K uses[-127,127].
GGML_BACKEND_API bool strata_iq3_dual_lut_q8_supported(const void * fullrow);
GGML_BACKEND_API bool strata_iq3_dual_lut_build_block(uint32_t * table,
    const void * q8rep0,const void * q8rep1,int block,int ith,int nth);
// rows=128, weight_stride_bytes=1760; q8a/b are original full Q8_K rows.
// Caller proves class0/1 match the table representatives on every d/qs block.
// Ny1: q8b/acc1/out1 allNULL; Ny2: allnonNULL, separate original output rows.
// acc0/1 contain rows*8 contiguous F32 lanes, out0/1 contain rows floats.
// block0 startszero,1..14 load/store lanes,15 writes original hsum to out only.
// Basic argument failure returnsfalse withoutwrites. No hot CPUID or inputscan;
// availability==7 and q8_supported are caller preconditions before any block.
GGML_BACKEND_API bool strata_iq3_dual_lut_chunk(size_t rows,const void * weight_row0,
    size_t weight_stride_bytes,const uint32_t * table,const void * q8a,const void * q8b,
    unsigned class0,unsigned class1,int block,float * acc0,float * acc1,float * out0,float * out1);
#else
// Unsupported platforms have no leaf link dependency and never mutate outputs.
static inline unsigned strata_iq3_dual_lut_capabilities(void) { return 0; }
static inline bool strata_iq3_dual_lut_q8_supported(const void * p) { (void)p;return false; }
static inline bool strata_iq3_dual_lut_build_block(uint32_t * t,const void * a,const void * b,int k,int i,int n) {
    (void)t;(void)a;(void)b;(void)k;(void)i;(void)n;return false;
}
static inline bool strata_iq3_dual_lut_chunk(size_t n,const void * w,size_t s,const uint32_t * t,
        const void * a,const void * b,unsigned c,unsigned d,int k,float * x,float * y,float * z,float * v) {
    (void)n;(void)w;(void)s;(void)t;(void)a;(void)b;(void)c;(void)d;(void)k;
    (void)x;(void)y;(void)z;(void)v;return false;
}
#endif
#ifdef __cplusplus
}
#endif
