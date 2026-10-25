#pragma once
#include "strata/qwen35/completed_v100/native/dense_gemm.hpp"
namespace strata::completed_v100::qwen35::native {
struct GdnPrefillProjection {Matrix matrix;float* output;};
bool try_nvfp4_gdn_prefill(DenseGemm*,const GdnPrefillProjection*,const float*,float*,size_t,int,cudaStream_t);
}
