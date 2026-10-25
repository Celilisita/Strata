#pragma once
#include <strata/qwen35/native/kv_ram.hpp>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strata::qwen35::native {
constexpr int DeviceSelectedCaches=17;
inline bool device_selected_parse(const char *p){
    if(!p||!std::strcmp(p,"0"))return false;
    if(!std::strcmp(p,"1"))return true;
    throw std::invalid_argument("STRATA_DENSE_DEVICE_SELECTED must be0or1");
}
inline bool device_selected_requested(){static const bool v=device_selected_parse(std::getenv("STRATA_DENSE_DEVICE_SELECTED"));return v;}
struct DeviceSelectedStats {bool requested=false,admitted=false;std::uint64_t caches=0,host_calls=0,saved_d2h_bytes=0,saved_summary_bytes=0;};
inline DeviceSelectedStats &device_selected_stats(){static thread_local DeviceSelectedStats s;return s;}
struct DeviceSelectedRole {
    std::array<const void*,DeviceSelectedCaches> key{},value{};
    std::uintptr_t owner=0;std::uint64_t epoch=0;bool active=false;
};
inline DeviceSelectedRole &device_selected_role(){static thread_local DeviceSelectedRole r;return r;}
inline bool device_selected_span(const void *p,size_t bytes,size_t alignment=4){
    const auto x=reinterpret_cast<std::uintptr_t>(p);
    return x&&x%alignment==0&&x<=std::numeric_limits<std::uintptr_t>::max()-bytes;
}
inline bool device_selected_overlap(const void *a,size_t an,const void *b,size_t bn){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<y+bn&&y<x+an;
}
inline bool device_selected_view_eligible(RamKvView v,int maximum){
    if(maximum<=32768||maximum>40957||maximum>v.host_capacity||v.host_capacity!=40960||v.resident!=40960||
       v.pages!=2560||v.slots!=2560||v.heads!=4||v.selected_tokens!=8192||v.selected_slots!=512||v.keep_queries!=3||
       v.host_key!=v.key||v.host_value!=v.value)return false;
    constexpr size_t payload=size_t(40960)*4*272,bounds=size_t(4)*2560*512*2,dir=size_t(4)*2560*4;
    const std::array<const void*,8> p{{v.key,v.value,v.bounds,v.page_slot,v.slot_page,v.sorted_ids,v.selected_slot,v.counters}};
    const std::array<size_t,8> n{{payload,payload,bounds,dir,dir,dir,dir,sizeof(RamKvCounters)}};
    for(size_t i=0;i<p.size();++i){if(!device_selected_span(p[i],n[i],i<3?16:4))return false;
        for(size_t j=0;j<i;++j)if(device_selected_overlap(p[i],n[i],p[j],n[j]))return false;}
    return v.page_ids&&v.offsets&&v.scores&&v.sorted_scores&&v.miss&&v.last_use&&v.epoch&&v.sort_work;
}
// Host request lease only: no RamKvView/CUDA parameter changes and no GPU allocation.
// Device authority starts with identity pages; all pages fit, so selection cannot evict one.
class DeviceSelectedRequestScope {
    std::array<RamKvCache*,DeviceSelectedCaches> cache_{};int count_=0,maximum_;bool eligible_;
public:
    DeviceSelectedRequestScope(bool phase_admitted,int maximum,bool sm120):maximum_(maximum),eligible_(phase_admitted&&sm120&&device_selected_requested()){
        device_selected_role()={};device_selected_stats()={};device_selected_stats().requested=device_selected_requested();
    }
    void add(RamKvCache &cache){
        if(count_==DeviceSelectedCaches){eligible_=false;return;}
        const auto v=cache.view();eligible_=eligible_&&cache.device_rows()&&device_selected_view_eligible(v,maximum_);
        for(int i=0;i<count_;++i){const auto old=cache_[i]->view();constexpr size_t bytes=size_t(40960)*4*272;
            if(device_selected_overlap(v.key,bytes,old.key,bytes)||device_selected_overlap(v.key,bytes,old.value,bytes)||
               device_selected_overlap(v.value,bytes,old.key,bytes)||device_selected_overlap(v.value,bytes,old.value,bytes))eligible_=false;}
        cache_[count_++]=&cache;
    }
    bool ready()const{return eligible_&&count_==DeviceSelectedCaches;}
    void keep(cudaStream_t stream){if(!ready())throw std::logic_error("device selected incomplete identity owner");
        cudaStreamCaptureStatus status{};
        if(cudaStreamIsCapturing(stream,&status)!=cudaSuccess||status!=cudaStreamCaptureStatusNone)
            throw std::logic_error("device selected keep before graph capture only");
        // Match the legacy demotion's directory/epoch reset, without moving encoded KV or summaries.
        for(auto *cache:cache_)cache->reset(stream);
    }
    void publish(){
        const auto &phase=phase_kv_stats();if(!ready()||!phase.migrated||phase.failed||phase.mapped_caches||phase.caches!=17)
            throw std::logic_error("device selected publication before full phase completion");
        auto &role=device_selected_role();role.owner=phase.owner;role.epoch=phase.epoch;role.active=true;
        for(int i=0;i<DeviceSelectedCaches;++i){role.key[i]=cache_[i]->view().key;role.value[i]=cache_[i]->view().value;}
        auto &s=device_selected_stats();s.admitted=true;s.caches=17;s.saved_d2h_bytes=1515192320ull;s.saved_summary_bytes=178257920ull;
    }
};
inline bool device_selected_attention(RamKvView v){
    const auto &r=device_selected_role();const auto &p=phase_kv_stats();
    if(!r.active||!p.migrated||p.failed||p.mapped_caches||p.owner!=r.owner||p.epoch!=r.epoch||!device_selected_view_eligible(v,40957))return false;
    for(int i=0;i<DeviceSelectedCaches;++i)if(v.key==r.key[i]&&v.value==r.value[i]){++device_selected_stats().host_calls;return true;}
    return false;
}
}
