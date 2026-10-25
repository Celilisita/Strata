#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
namespace strata::qwen35::native {
void nvfp4_warp_logical_linear(Nvfp4View,const void*,float*,int,cudaStream_t);
void nvfp4_warp_logical_pair(Nvfp4View,Nvfp4View,const void*,float*,float*,float*,int,cudaStream_t);
}
