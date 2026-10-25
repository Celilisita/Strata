#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
namespace strata::qwen35::native {
void nvfp4_swiglu_vector(const float*,const float*,float*,Nvfp4Input,int,int,float,cudaStream_t);
}
