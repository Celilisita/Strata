#pragma once
#include "strata/qwen35/completed_v100/native/attention_prefill.hpp"
#include "strata/qwen35/completed_v100/native/attention_decode.hpp"

namespace strata::completed_v100::qwen35::native {
// Stable graph grid: head x KV partitions, with the live causal extent read
// from graph_start. Only partition numerators/denominators occupy global scratch.
int verify_attention_partitions(int maximum_valid,int heads,DecodeRounding);
size_t verify_attention_workspace(PrefillAttentionShape,int partitions);
// Common partial layout/merge; callers keep scratch alive to stream completion.
void verify_attention_merge(const float *,float *,int heads,int tokens,int partitions,cudaStream_t);
void verify_attention_q8(const float *query,const void *key,const void *value,
                         float *output,void *scratch,size_t scratch_bytes,
                         PrefillAttentionShape,int partitions,cudaStream_t);
}
