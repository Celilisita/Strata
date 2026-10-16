#pragma once
#include <cuda_runtime.h>

namespace strata::qwen35::native {
__device__ inline float swiglu_value(float gate, float up) {
    const float denominator = __fadd_rn(1.0f, __expf(-gate));
    return __fmul_rn(__fdividef(gate, denominator), up);
}
}
