#pragma once
#include "strata/qwen35/completed_v100/native/gdn.hpp"
namespace strata::completed_v100::qwen35::native::prefetch_gdn {
namespace n=strata::completed_v100::qwen35::native;
void run(n::GdnInputs,n::GdnShape,float*,float*,void*,size_t,cudaStream_t,void(*)(void*,cudaStream_t),void*);
}
