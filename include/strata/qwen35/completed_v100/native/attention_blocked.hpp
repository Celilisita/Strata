#pragma once
#include "strata/qwen35/completed_v100/native/attention_gemm.hpp"

namespace strata::completed_v100::qwen35::native {
// Query-bounded FP32 score arena; no full-context quadratic allocation.
size_t blocked_attention_workspace(PrefillAttentionShape,int query_rows=128);
void blocked_attention(DenseGemm &,const float *,const void *,const void *,float *,
                       void *,size_t,PrefillAttentionShape,int query_rows,cudaStream_t);
}
