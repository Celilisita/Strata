#pragma once
#include "strata/qwen35/completed_v100/native/gdn.hpp"
namespace strata::completed_v100::qwen35::native {
void gdn_recurrence_subwarp8_dual(GdnInputs,GdnShape,float*,float*,void*,size_t,cudaStream_t);
}
