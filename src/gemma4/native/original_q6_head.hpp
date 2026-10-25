#pragma once
#include "strata/gemma4/native/decoder_ops.hpp"
namespace strata::gemma4::native {
bool original_q6_head_requested();
// Fixed tied raw-Q6/F32-input leaf: no activation re-encoding or owned allocation.
void original_q6_head(primitive::Matrix,const float*,float*,cudaStream_t);
// Contiguous 2/3 target features; share raw weights without changing row math.
void original_q6_head_rows(primitive::Matrix,const float*,float*,int,cudaStream_t);
}
