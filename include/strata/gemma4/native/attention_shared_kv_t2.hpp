#pragma once
#include "strata/gemma4/native/attention_parallel_merge.hpp"

namespace strata::gemma4::native::shared336 {
constexpr bool DefaultEnabled=false;
// Global Q32/KV4/D512/window0/T2 only, capacity2..32768. One head/part CTA
// shares K/V decode across two causal queries; their online states are separate.
// Same full-capacity AoS [query][head][part][m,z,numerator512] as frozen321.
// Future parts are written as [-Inf,0,zeros]; an invalid query writes all NaNs.
// No allocation/copy/sync; merge is the unchanged frozen321 implementation.
void split_attention_shared_kv_t2(const float* query,float* output,CacheView,
    const int* position,int heads,int tokens,merge321::Lease,merge321::Binding live);
}
