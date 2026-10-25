#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
#include <cstddef>
namespace strata::qwen35::native {
// Task-only PP SSM output: dead F32 output is validated but never written.
struct GdnOutputElisionResources {int registers=0,blocks=0,binary_version=0;size_t shared_bytes=0,local_bytes=0;};
GdnOutputElisionResources gdn_output_elision_resources();
void gdn_output_elision_validate(const float *attention,const float *gamma,const float *gate,float *gated,
    Nvfp4Input encoded,int tokens,float epsilon,float global);
void gdn_output_elision_encode(const float *attention,const float *gamma,const float *gate,float *gated,
    Nvfp4Input encoded,int tokens,float epsilon,float global,cudaStream_t stream);
}
