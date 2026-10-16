#pragma once
#include "expert_fill_upload.hpp"
#include "cache_vmm_phase.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
namespace strata::deepseek_v4 {
// Quiescent phase transitions only. Authoritative host weights rebuild a
// changed bank without a second full host/GPU copy of the old cache.
template<class Impl,class Upload> size_t resize_phase_cache(Impl& p,int permanent,Upload upload) {
    if(p.groups.size()!=1)throw std::runtime_error("phase cache requires one residency authority");
    auto& g=p.groups[0];auto& d=p.devices[g.id];auto& lru=g.lru;
    if(!d.phase_banks.empty())return resize_vmm_cache(p,permanent);
    const int old=lru.n_slots;
    if(permanent==old)return 0;
    if(permanent<6||permanent>int(g.layers.size())*lru.n_expert)throw std::runtime_error("phase cache capacity outside model");
    std::vector<int32_t> keys=lru.key_of;
    if(permanent<old){
        std::stable_sort(keys.begin(),keys.end(),[&](int a,int b){
            return a>=0&&(b<0||(lru.frequency&&lru.frequency->score(a)>lru.frequency->score(b)));});
        keys.resize(permanent);
    }else keys.resize(permanent,-1);
    ggml_backend_synchronize(d.backend);
    d.buf.reset();
    for(auto* t=ggml_get_first_tensor(d.ctx.get());t;t=ggml_get_next_tensor(d.ctx.get(),t)){t->data=nullptr;t->buffer=nullptr;}
    g.n_slots=permanent+g.native.slots;
    for(auto* bank:g.banks){bank->ne[2]=g.n_slots+1;bank->nb[3]=bank->nb[2]*bank->ne[2];}
    for(auto& entry:p.bindings){auto* t=entry.second.cached;t->ne[2]=g.n_slots;t->nb[3]=t->nb[2]*t->ne[2];}
    d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(),d.buft));
    if(!d.buf)throw std::runtime_error("resized phase expert cache allocation failed");
    d.buf_size=ggml_backend_buffer_get_size(d.buf.get());
    ggml_backend_buffer_set_usage(d.buf.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for(auto* map:lru.slot_map)if(map)std::fill_n(map,lru.n_expert,-1);
    lru.n_slots=permanent;lru.key_of=keys;lru.prev.resize(permanent);lru.next.resize(permanent);
    for(int slot=0;slot<permanent;++slot){lru.prev[slot]=slot-1;lru.next[slot]=slot+1<permanent?slot+1:-1;}
    lru.head=0;lru.tail=permanent-1;g.native.keys.fill(-1);
    size_t bytes=0;
    for(int slot=0;slot<permanent;++slot)if(keys[slot]>=0){
        const int layer=keys[slot]/lru.n_expert,id=keys[slot]%lru.n_expert;
        lru.slot_map[layer][id]=slot;
        for(size_t role=0;role<g.banks.size();++role){
            const auto* source=p.layers[layer].experts[role];auto* bank=g.banks[role];
            upload(d.backend,bank,static_cast<const char*>(source->data)+id*source->nb[2],slot*bank->nb[2],bank->nb[2]);
            bytes+=bank->nb[2];
        }
    }
    return bytes;
}
// Reserve a minimum quota per layer, then distribute remaining slots to the
// largest next marginal prefill frequency. No second expert directory exists.
template<class Impl,class Upload> size_t prewarm_phase_cache(Impl& p,Upload upload) {
    auto& g=p.groups.at(0);auto& lru=g.lru;
    if(!lru.frequency)return 0;
    std::vector<std::vector<int>> ranked(g.layers.size());
    std::vector<int> quota(g.layers.size(),0),selected;
    for(size_t il=0;il<ranked.size();++il){auto& r=ranked[il];r.resize(lru.n_expert);std::iota(r.begin(),r.end(),0);
        std::stable_sort(r.begin(),r.end(),[&](int a,int b){return lru.frequency->score(g.layers[il]*lru.n_expert+a)>lru.frequency->score(g.layers[il]*lru.n_expert+b);});}
    const int usable=std::max(0,lru.n_slots-6); // Six unused rows permit asynchronous admission without overwriting readers.
    const int minimum=std::min(6,usable/int(ranked.size()));
    for(size_t il=0;il<ranked.size();++il)for(int n=0;n<minimum;++n){selected.push_back(g.layers[il]*lru.n_expert+ranked[il][quota[il]++]);}
    while(int(selected.size())<usable){int best=-1;uint32_t score=0;
        for(size_t il=0;il<ranked.size();++il)if(quota[il]<lru.n_expert){const auto s=lru.frequency->score(g.layers[il]*lru.n_expert+ranked[il][quota[il]]);
            if(best<0||s>score||(s==score&&quota[il]<quota[best])){best=int(il);score=s;}}
        if(best<0)break;selected.push_back(g.layers[best]*lru.n_expert+ranked[best][quota[best]++]);}
    std::vector<bool> wanted(lru.slot_map.size()*lru.n_expert,false);for(int key:selected)wanted[key]=true;
    for(int slot=0;slot<lru.n_slots;++slot){const int key=lru.key_of[slot];if(key>=0&&!wanted[key]){lru.slot_map[key/lru.n_expert][key%lru.n_expert]=-1;lru.key_of[slot]=-1;}}
    size_t bytes=0;int free=0;
    for(int key:selected)if(lru.slot_map[key/lru.n_expert][key%lru.n_expert]<0){
        while(free<lru.n_slots&&lru.key_of[free]>=0)++free;
        if(free==lru.n_slots)throw std::runtime_error("prefill quota exceeds cache slots");
        lru.key_of[free]=key;lru.slot_map[key/lru.n_expert][key%lru.n_expert]=free;
        for(size_t role=0;role<g.banks.size();++role){const auto* source=p.layers[key/lru.n_expert].experts[role];auto* bank=g.banks[role];
            upload(p.devices[g.id].backend,bank,static_cast<const char*>(source->data)+(key%lru.n_expert)*source->nb[2],free*bank->nb[2],bank->nb[2]);bytes+=bank->nb[2];}
    }
    return bytes;
}
}
