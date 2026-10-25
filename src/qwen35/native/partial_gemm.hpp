#pragma once
#include "strata/qwen35/native/dense_gemm.hpp"
namespace strata::qwen35::native {
std::unique_ptr<DenseGemm> make_partial16_gemm(DenseGemm &,cudaStream_t);
}
