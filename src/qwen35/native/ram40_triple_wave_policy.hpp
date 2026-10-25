#pragma once
#include "ram_batch_select_policy.hpp"
namespace strata::qwen35::native {
constexpr size_t TriplePartialOffset=2740352+128,TriplePartialBytes=size_t(24)*64*3*258*4;
constexpr size_t TripleScoreOffset=TriplePartialOffset+TriplePartialBytes+128,TripleScoreBytes=size_t(3)*4*64*6*128*4;
constexpr size_t TripleBorrowedBytes=TripleScoreOffset+TripleScoreBytes+128;
static_assert(TripleBorrowedBytes==9855488&&TripleBorrowedBytes<356515840,"triple-wave bounded dead FFN lease");
struct Ram40TripleWaveResources {int registers,blocks_per_sm,shared_bytes,local_bytes,binary_version;};
Ram40TripleWaveResources ram40_triple_wave_resources();
struct Ram40TripleWaveStats {
    bool requested=false,admitted=false,role_admitted=false;
    std::uint64_t host_calls=0,fallback_calls=0,epoch_bound=0,phase_epoch=0;std::uintptr_t owner=0;
    int last_tokens=0,last_pages=0;size_t borrowed_bytes=0;Ram40TripleWaveResources resource{};
};
inline Ram40TripleWaveStats& ram40_triple_wave_state(){static thread_local Ram40TripleWaveStats s;return s;}
inline Ram40TripleWaveStats ram40_triple_wave_stats(){return ram40_triple_wave_state();}
inline bool ram40_triple_wave_requested(){
    static const bool value=[](){const char*p=std::getenv("STRATA_DENSE_RAM40_TRIPLE_WAVE");
        if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("RAM40 triple-wave flag must be0or1");return p&&p[0]=='1';}();return value;
}
// PREPARE only, after graph release. Never query CUDA resources inside capture.
inline void configure_ram40_triple_wave_request(bool allowed){
    auto&s=ram40_triple_wave_state();s={};s.requested=ram40_triple_wave_requested();if(!s.requested||!allowed)return;
    int device=0;check_cuda(cudaGetDevice(&device),"triple-wave cold device");const auto c=capabilities(device);
    if(c.major!=12||c.minor!=0)return;s.resource=ram40_triple_wave_resources();
    if(s.resource.binary_version!=120||s.resource.local_bytes||s.resource.blocks_per_sm<1)throw std::runtime_error("triple-wave resource gate");
    s.admitted=true;
}
inline bool use_ram40_triple_wave(bool batch,RamKvView v,int tokens,int heads,int parts,int shift,size_t bytes){
    auto&s=ram40_triple_wave_state();if(!s.admitted||!batch)return false;
    const auto b=ram_batch_select_stats();
    // The existing batch-selector has just admitted this exact query/owner/epoch.
    // Do not re-run its predicate: doing so would double-count selector submissions.
    const bool use=b.role_admitted&&b.owner&&b.phase_epoch&&b.last_tokens==3&&b.last_pages==2560&&
        tokens==3&&heads==24&&parts==64&&!shift&&v.pages==2560&&v.slots==2560&&v.resident==40960&&
        v.host_capacity==40960&&v.heads==4&&v.selected_tokens==8192&&v.selected_slots==512&&v.keep_queries==3&&
        v.host_key==v.key&&v.host_value==v.value&&bytes>=TripleBorrowedBytes;
    if(use){s.role_admitted=true;++s.host_calls;s.epoch_bound=b.epoch_bound;s.phase_epoch=b.phase_epoch;s.owner=b.owner;
        s.last_tokens=3;s.last_pages=2560;s.borrowed_bytes=TripleBorrowedBytes;}
    else ++s.fallback_calls;return use;
}
}
