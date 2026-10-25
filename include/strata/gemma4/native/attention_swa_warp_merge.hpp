#pragma once
#include "strata/gemma4/native/decoder_ops.hpp"
#include <cstdint>
namespace strata::gemma4::native {
struct SwaWarpMergeT1Binding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct SwaWarpMergeT1Lease {
    void* data=nullptr;size_t bytes=0;SwaWarpMergeT1Binding binding;int max_position_exclusive=0;
};
// Fixed localD256/Q32/KV16/window1024, 174 AoS32part scratch (32keys perpart).
size_t swa_warp_merge_t1_bytes(CacheView,int heads);
// Research/defaultunregistered: same174 perkey online math/4warp combine andabsolute ring,
// Partition32 andwarpD32 merge; caller keeps borrowed arena/owner/epoch live.
void swa_warp_merge_t1(const float*,float*,CacheView,const int* device_position,
    int heads,int tokens,SwaWarpMergeT1Lease,SwaWarpMergeT1Binding live);
}
