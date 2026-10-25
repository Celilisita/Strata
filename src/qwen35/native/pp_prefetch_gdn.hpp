#pragma once
#include "strata/qwen35/native/gdn.hpp"
namespace strata::qwen35::native::prefetch_gdn {
namespace n=strata::qwen35::native;
void run(n::GdnInputs,n::GdnShape,float*,float*,void*,size_t,cudaStream_t,void(*)(void*,cudaStream_t),void*);
}
