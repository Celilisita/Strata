#pragma once
#include "ggml-backend.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API int strata_glm_cpu_row_dot_available(void);
// Original IQ3_S rows/stride and canonical Q8_K activation(s). Low32 return
// bits count actual double-row calls; high32 count original-dot fallback rows.
GGML_BACKEND_API uint64_t strata_glm_cpu_row_dot(int n,float* out0,float* out1,
    const void* rows,size_t stride,const void* activation0,const void* activation1,int count);
#ifdef __cplusplus
}
#endif
