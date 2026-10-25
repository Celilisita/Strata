#pragma once
#include "strata/qwen35/completed_v100/native/small_prefill.hpp"

namespace strata::completed_v100::qwen35::native {
// Experimental detector never changes rounding or quantized codes. It marks
// whole32-value groups for an independent owned projection recomputation.
void small_prefill_q4_boundary_mask(const float *,unsigned *,int,int,cudaStream_t);
void small_prefill_q4_masked_linear(Matrix,const void *,float *,const unsigned *,
                                   int,cudaStream_t,int);
}
