#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cstddef>
namespace strata::gemma4::native {
// Explicit cold opt-in; the default retains the original F32 tiled provider.
bool pp_tc_requested();
std::size_t pp_tc_workspace_bytes(int maximum_tokens);
// Original packed Q8 cache, raw Gemma normalized/RoPE F32 Q, scale exactly1.
// Output is the online F32 accumulator; the owning service keeps it disjoint.
void pp_tc_attention(const float*,float*,CacheView,const int*,int past,int tokens,
                     int maximum_tokens,AttentionWorkspace,cudaStream_t);
}
