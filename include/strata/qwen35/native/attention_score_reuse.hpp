#pragma once
#include "strata/qwen35/native/attention_verify_grouped.hpp"
namespace strata::qwen35::native {
struct VerifyScoreReuseLayout {
    size_t partial_bytes,score_offset,score_bytes,total_bytes;
    int keys_per_partition;
};
// Narrow direct-resident Q8 contract: T3,24Q/4KV,dim256,32parts,capacity<=16K.
// Scratch is the existing dead decoded-weight arena, borrowed through ordered
// stream completion. It cannot overlap live Q/K/V/output/cursor allocations.
VerifyScoreReuseLayout verify_score_reuse_layout(PrefillAttentionShape,int partitions);
VerifyGroupedResources verify_grouped_score_resources();
void verify_attention_grouped_score_q8(const float *,const void *,const void *,float *,void *,size_t,
                                      PrefillAttentionShape,int partitions,cudaStream_t);
}
