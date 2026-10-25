#pragma once
#include "ggml-backend.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API int strata_glm_cpu_iq4_packed_available(void);
GGML_BACKEND_API void strata_glm_cpu_iq4_packed_dot(int,float*,size_t,const void*,size_t,const void*,size_t,int);
#ifdef __cplusplus
}
#endif
