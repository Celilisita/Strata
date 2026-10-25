#pragma once
#include "strata/gemma4/native/attention_global_aos_merge.hpp"
namespace strata::gemma4::native {
// Research-only T1/global/D512/Q32/KV4. Original AoS [m,z,numerator512].
// Eight warps share a four-key F32 K/V tile; no persistent decodedKV cache.
// Same owned Ffn-workspace/epoch/stream lease as266; default unregistered.
inline AosMergeT1Layout global_gqa_shared_t1_layout(CacheView c,int heads,int context){
    return global_parallel_aos_merge_t1_layout(c,heads,context);
}
void global_gqa_shared_t1(const float*,float*,CacheView,const int*,int heads,int tokens,
    AosMergeT1Lease,AosMergeT1Binding live);
}
