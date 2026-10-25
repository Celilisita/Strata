#pragma once
#include "strata/qwen35/completed_v100/native/quant.hpp"
#include <cuda_runtime.h>

namespace strata::completed_v100::qwen35::native {
// Original GGUF Q8_0 blocks; float normalized input [T,K], outputs [T,R].
// One fused submission, no activation re-quantization or weight expansion.
void q8_projection_pair(Matrix alpha, Matrix beta, const float* input,
                        float* a, float* b, int tokens, cudaStream_t stream);
}
