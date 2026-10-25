#pragma once
#include "strata/qwen35/completed_v100/native/attention_verify.hpp"
namespace strata::completed_v100::qwen35::native::q36_online {
// Existing Hadamard Q and complete contiguous GPU Q8 KV; no second owner/cache.
// KV[token][4][8 blocks of 34 bytes], Q[token][24][256], scale=1/16.
// Caller keeps Q/K/V/output/scratch and graph_start live through all Graphs.
// graph_start is the absolute live T3 base, already range-validated by MTP.
// Same 258-float (numerator256,max,denom) partial ABI and parent merge.
void configure(); // cold, after old Graphs are released, before capture
void attention(const float*,const void*,const void*,float*,void*,size_t,
               PrefillAttentionShape,int partitions,cudaStream_t);
}
