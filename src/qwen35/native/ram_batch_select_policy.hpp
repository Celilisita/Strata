#pragma once
#include "device_selected_request.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <climits>
namespace strata::qwen35::native {
struct RamBatchSelectResources {int score_registers,sort_registers,score_blocks,sort_blocks,local_bytes,binary_version;};
RamBatchSelectResources ram_batch_select_resources();
struct RamBatchSelectStats {
    bool requested=false,admitted=false,role_admitted=false;std::uint64_t host_calls=0,fallback_calls=0,epoch_bound=0,phase_epoch=0;
    std::uintptr_t owner=0;int last_tokens=0,last_pages=0;
};
inline RamBatchSelectStats& ram_batch_select_state(){static thread_local RamBatchSelectStats s;return s;}
inline RamBatchSelectStats ram_batch_select_stats(){return ram_batch_select_state();}
inline bool ram_batch_select_requested(){
    static const bool value=[](){const char*p=std::getenv("STRATA_DENSE_RAM_BATCH_SELECT");
        if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("RAM batch selector flag must be0or1");return p&&p[0]=='1';}();return value;
}
// PREPARE after Graph release; resource queries never run during graph capture.
inline void configure_ram_batch_select_request(bool allowed){
    auto&s=ram_batch_select_state();s={};s.requested=ram_batch_select_requested();if(!s.requested||!allowed)return;
    int device=0;check_cuda(cudaGetDevice(&device),"batch-select cold device");const auto c=capabilities(device);
    if(c.major!=12||c.minor!=0)return;const auto r=ram_batch_select_resources();
    if(r.binary_version!=120||r.local_bytes||r.score_blocks<1||r.sort_blocks<1)throw std::runtime_error("batch-select resource gate");
    s.admitted=true;
}
inline bool use_ram_batch_select(RamKvView v,int tokens,int heads,int parts,int shift,size_t bytes){
    auto&s=ram_batch_select_state();if(!s.admitted)return false;
    const auto&r=device_selected_role();const auto&p=phase_kv_stats();bool owned=false;
    for(int i=0;i<DeviceSelectedCaches;++i)owned=owned||(v.key==r.key[i]&&v.value==r.value[i]);
    // This published role exists only after all17 identity-directory/epoch resets
    // have joined. All pages fit: old T1/T2 planners cannot evict or remap them.
    // Decode positions grow within pp_bound; <=3 queries per position keeps the
    // per-head int epoch below 3*40960, without a per-round D2H or synchronization.
    static_assert(3*40960<INT_MAX-3,"finite request selection epoch");
    const bool use=r.active&&owned&&p.migrated&&!p.failed&&p.caches==17&&p.main_caches==16&&p.draft_caches==1&&!p.mapped_caches&&p.owner&&p.epoch&&
        p.owner==r.owner&&p.epoch==r.epoch&&p.pp_bound>32768&&p.pp_bound<=40960&&
        device_selected_view_eligible(v,40957)&&tokens==3&&heads==24&&parts==64&&!shift&&bytes>=2740352;
    if(use){s.role_admitted=true;s.epoch_bound=3ull*p.pp_bound;s.phase_epoch=p.epoch;s.owner=p.owner;
        ++s.host_calls;s.last_tokens=tokens;s.last_pages=v.pages;}
    else ++s.fallback_calls;return use;
}
}
