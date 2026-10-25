#pragma once
#include "q6_soa.hpp"
namespace strata::qwen35::native {
// Host policy is changed only by an idle, drained request controller.
int q6_head_warps();
void set_q6_head_warps(int warps);
// Same080 encoded I8/F16 operands; only reduction/row geometry changes.
void project_q6_head(Matrix,const void*,float*,int tokens,int warps,cudaStream_t);
}
