#pragma once
#include "strata/gemma4/native/vector3.hpp"

namespace strata::gemma4::native {
// Explicit, unregistered T1 Q4_K experiment. Prepared input is the existing
// ResidualVectorBlockBytes=120 layout per32 values, never Single40.
// No quantization, K ordering, accumulation, or Model policy is changed.
void linear_vector_grouped_aligned_q4_t1(primitive::Matrix, const void* prepared,
    float* output, int tokens, cudaStream_t);
}
