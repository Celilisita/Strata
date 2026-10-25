#pragma once
#include "pp_tc_attention.hpp"
namespace strata::gemma4::native::global_keys1024 {
// Declaration-only bridge to the exact qualified diagnostic leaf objects.
bool pp_tc_requested();
std::size_t pp_tc_workspace_bytes(int maximum_tokens);
void pp_tc_attention(const float*,float*,CacheView,const int*,int past,int tokens,
                     int maximum_tokens,AttentionWorkspace,cudaStream_t);
}
