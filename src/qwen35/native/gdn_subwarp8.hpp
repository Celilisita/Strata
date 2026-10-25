#pragma once
#include "strata/qwen35/native/gdn.hpp"
namespace strata::qwen35::native {
void gdn_recurrence_subwarp8_dual(GdnInputs,GdnShape,float*,float*,void*,size_t,cudaStream_t);
}
