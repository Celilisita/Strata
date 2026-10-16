#pragma once
#include "strata/prefill/gemm.hpp"

namespace strata::prefill {
// Benchmark-only symmetric candidates. Stateful use requires route gates;
// affine formats additionally require canonical scale/min validation.
bool precise_dense_candidate(int type) noexcept;
// F32 dequantization semantics, represented as high + low/4096 in F16.
// Caller owns bounded scratch; three Tensor Core terms omit only low*low.
void precise_native(Gemm& gemm, const uint16_t* input_high, const uint16_t* input_low,
                    int type, const void* weights, float* output,
                    int64_t tokens, int64_t rows, int64_t cols,
                    float* weight_f32, uint16_t* weight_low, int64_t scratch_elements);
}
