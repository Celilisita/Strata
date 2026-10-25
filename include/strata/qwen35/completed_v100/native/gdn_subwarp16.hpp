#pragma once
#include "strata/qwen35/completed_v100/native/gdn.hpp"
namespace strata::completed_v100::qwen35::native {
size_t gdn_subwarp16_workspace_bytes(GdnShape);
void gdn_recurrence_subwarp16(GdnInputs,GdnShape,float*,float*,void*,size_t,cudaStream_t);
void gdn_recurrence_subwarp16_dual(GdnInputs,GdnShape,float*,float*,void*,size_t,cudaStream_t);
}
