#pragma once
#include "strata/gemma4/native/vector_single.hpp"

namespace strata::gemma4::native {
// Explicit opt-in research API; no Model or default dispatch is registered.
// Same Single40 input and raw Q4_K payload as 202. T1 only; Q6 is rejected.
void linear_single_q4_striped_t1(primitive::Matrix, SingleQ8View, float*, cudaStream_t);
void pair_single_q4_striped_t1(primitive::Matrix, primitive::Matrix, SingleQ8View,
    float* gate, float* up, float* activation, cudaStream_t);
}
