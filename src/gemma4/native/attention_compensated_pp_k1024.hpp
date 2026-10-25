#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include "attention_compensated_pp_k1024_geometry.hpp"
#include <cstdint>
namespace strata::gemma4::native {
struct CompensatedPpK1024Binding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct CompensatedPpK1024Lease {
    void* dead_k=nullptr;std::size_t dead_k_bytes=0;
    void* dead_v=nullptr;std::size_t dead_v_bytes=0;
    CompensatedPpK1024Binding binding;
};
// Independent/globalD512/Q32/KV4/GQA8 PP candidate; no Model hook.
// GPUposition==hostpast. Cache[0,past+tokens) is validQ8 andafterend untouched.
// K/Q/P havehighplane followedbylowplane; l=RN16((x-RN16(x))*4096).
// Vhigh/low eachbank contains2physicalheads, in dead_k (0/1),dead_v(2/3).
void compensated_physical_pp_k1024_attention(const float*,float*,CacheView,const int*,
    int past,int heads,int tokens,AttentionWorkspace,CompensatedPpK1024Lease,CompensatedPpK1024Binding live);
}
