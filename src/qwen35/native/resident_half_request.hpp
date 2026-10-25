#pragma once
#include "strata/qwen35/native/dense_gemm.hpp"
namespace strata::qwen35::native {
void nvfp4_configure_half_request(DenseGemm &,bool resident,int prompt);
unsigned long long nvfp4_half_submissions(DenseGemm &);
}
