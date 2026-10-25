#pragma once
#include "ggml-backend.h"
#include <stdint.h>
#include <stdbool.h>
// Pure CPUID/OS decoder shared with bounded host tests. Caller supplies CPUID.
static inline unsigned strata_glm_cpu_vnni_decode(unsigned leaf1c,unsigned leaf7b,unsigned leaf7c,uint64_t xcr0) {
    const unsigned base=(1u<<12)|(1u<<28)|(1u<<29);
    const unsigned cpu=(1u<<5)|(1u<<16)|(1u<<17)|(1u<<30)|(1u<<31);
    const bool hardware=(leaf1c&base)==base&&(leaf7b&cpu)==cpu&&(leaf7c&(1u<<11));
    const bool os=(leaf1c&0x1c000000u)==0x1c000000u&&(xcr0&0xe6u)==0xe6u;
    return 1u|(hardware?2u:0u)|(os?4u:0u);
}
#ifdef __cplusplus
extern "C" {
#endif
// Entry points require capabilities==7 at the graph-caller dispatch.
// Bit0 compiled IQ3_S variants; bit1 hardware CPUID; bit2 OS XCR0 state.
GGML_BACKEND_API unsigned strata_glm_cpu_vnni_capabilities(void);
GGML_BACKEND_API void strata_glm_cpu_iq3_s_vnni_pair(int,float*,float*,const void*,const void*,const void*);
GGML_BACKEND_API void strata_glm_cpu_iq3_s_vnni(int,float*,size_t,const void*,size_t,const void*,size_t,int);
#ifdef __cplusplus
}
#endif
