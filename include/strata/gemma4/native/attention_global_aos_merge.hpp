#pragma once
#include "strata/gemma4/native/decoder_ops.hpp"
#include <cstdint>
namespace strata::gemma4::native {
struct AosMergeT1Binding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct AosMergeT1Lease {AttentionWorkspace workspace{};AosMergeT1Binding binding;int max_position_exclusive=0;};
struct AosMergeT1Layout {int parts=0;size_t bytes=0;};
// Original174 AoS row: [m,z,numerator512], partition128; no conversion buffer.
inline AosMergeT1Layout global_parallel_aos_merge_t1_layout(CacheView cache,int heads,int context){
    if(heads!=32||cache.heads!=4||cache.dim!=512||cache.window!=0||cache.capacity<1||cache.capacity>262144||
       context<1||context>cache.capacity)throw std::invalid_argument("Gemma AoS merge T1 global geometry");
    const int parts=(cache.capacity+127)/128;
    return {parts,size_t(32)*parts*514*sizeof(float)};
}
// Research API, default unregistered; lends existing dead Ffn decode arena.
// DeviceP bounds/active parts are checked on GPU, including rewind and invalidP.
void global_parallel_aos_merge_t1(const float*,float*,CacheView,const int*,int heads,int tokens,
    AosMergeT1Lease,AosMergeT1Binding live);
}
