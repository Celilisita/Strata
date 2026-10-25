#pragma once
#include "strata/qwen35/native/ffn.hpp"
namespace strata::qwen35::native {
void q6_warp_logical(Matrix,const void*,float*,int,cudaStream_t);
}
