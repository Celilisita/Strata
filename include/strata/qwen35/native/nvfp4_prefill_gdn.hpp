#pragma once
#include "strata/qwen35/native/dense_gemm.hpp"
namespace strata::qwen35::native {
struct GdnPrefillProjection {Matrix matrix;float* output;};
bool try_nvfp4_gdn_prefill(DenseGemm*,const GdnPrefillProjection*,const float*,float*,size_t,int,cudaStream_t);
}
