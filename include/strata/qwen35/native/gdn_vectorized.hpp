#pragma once
#include "strata/qwen35/native/gdn.hpp"
#include <cstddef>

namespace strata::qwen35::native {
// Borrowed transient FP32 layout; no persistent state or hot allocation.
size_t gdn_vectorized_workspace_bytes(GdnShape);
void gdn_recurrence_vectorized(GdnInputs,GdnShape,float *,float *,void *,size_t,cudaStream_t);
}
