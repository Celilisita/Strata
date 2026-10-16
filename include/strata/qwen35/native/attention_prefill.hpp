#pragma once
#include <cuda_runtime.h>

namespace strata::qwen35::native {
// FP16 rounding intervals are explicit numeric policy, independent of CTA placement.
// Must be calibrated to the caller's checked model/precision contract.
struct PrefillRounding { int query_group=32,head_group=2,segments=0,padded_keys=0; };
struct PrefillAttentionShape {
    int dim, query_heads, kv_heads, tokens, query_start, valid, capacity;
    float scale;
    PrefillRounding rounding;
    bool q8=false;
    const int *graph_start=nullptr;
};
// Q[tokens,heads,256], Q4_0 KV[capacity,kv_heads,8,18]. Ordinary causal text.
// Own FP16 operands, F32 QK/normalizer, FP16 value accumulator contract.
// Decode each valid Q4 KV element once per invocation into borrowed half staging.
// The workspace is reusable and may alias dead projection weight staging.
size_t prefill_attention_workspace(PrefillAttentionShape);
void prefill_decode_kv(const void *k,const void *v,void *workspace,size_t bytes,PrefillAttentionShape,cudaStream_t);
void prefill_attention(const float *q, const void *k, const void *v, float *output, void *workspace, size_t workspace_bytes,
                       PrefillAttentionShape, cudaStream_t);
}
