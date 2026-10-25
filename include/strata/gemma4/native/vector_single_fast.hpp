#pragma once
#include "strata/gemma4/native/vector_single.hpp"

namespace strata::gemma4::native {
// Same SingleQ8View/payload and reduction as 196. Q4 GGUF code/header reads
// use their proven four-byte alignment; Q6 delegates to the original path.
void linear_single_q8_fast(primitive::Matrix,SingleQ8View,float*,cudaStream_t);
void pair_single_q8_fast(primitive::Matrix,primitive::Matrix,SingleQ8View,
    float* gate,float* up,float* activation,cudaStream_t);
}
