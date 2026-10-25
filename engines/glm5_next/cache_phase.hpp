#pragma once
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "phase_options.hpp"
#include "cache_budget.hpp"
#include "cache_inventory.hpp"
#include "../deepseek_v4/cache_vmm_phase.hpp"
#include <numeric>
#include <cstring>

namespace strata::glm5_next {
// Mutate the existing cache groups only after their readers and graphs retire.
// Stable VMM addresses keep all bank views attached to the same ownership.
template<class Impl,class Device>
void allocate_phase_banks(Impl& p,Device& d) {
    if(p.devices.size()!=1 || (p.groups.size()!=3 && p.groups.size()!=4))
        throw std::runtime_error("GLM phase cache requires the verified three/four-layout single-device inventory");
    auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
    auto create=reinterpret_cast<deepseek_v4::PhaseCreate>(ggml_backend_reg_get_proc_address(reg,"strata_phase_buffer_new"));
    if(!create)throw std::runtime_error("GLM phase VMM factory unavailable");
    d.buf_size=0;
    for(auto& group:p.groups) {
        if(group.banks.size()!=3 || group.lru.n_expert!=288 || group.native.slots)
            throw std::runtime_error("GLM phase bank geometry differs");
        for(auto* bank:group.banks) {
            auto* buffer=create(d.backend,ggml_nbytes(bank),bank->nb[2]*(group.layers.size()*288+1));
            if(!buffer)throw std::runtime_error("GLM phase bank mapping failed");
            d.phase_banks.emplace_back(buffer);
            ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            if(ggml_backend_tensor_alloc(buffer,bank,ggml_backend_buffer_get_base(buffer))!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("GLM phase bank attachment failed");
            d.buf_size+=ggml_backend_buffer_get_size(buffer);
        }
    }
    for(auto& entry:p.bindings)
        if(ggml_backend_view_init(entry.second.cached)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("GLM phase borrowed view attachment failed");
}

template<class Impl> void resize_phase(Impl& p,size_t budget) {
    auto& d=p.devices.at(0);
    if(d.phase_banks.size()!=3*p.groups.size())throw std::runtime_error("GLM phase owner bank count differs");
    auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
    auto resize=reinterpret_cast<deepseek_v4::PhaseResize>(ggml_backend_reg_get_proc_address(reg,"strata_phase_buffer_resize"));
    using Swap=void(*)(ggml_backend_t,void*,size_t,void*,size_t,void*,size_t,int,int);
    auto swap=reinterpret_cast<Swap>(ggml_backend_reg_get_proc_address(reg,"strata_phase_swap_rows_mixed"));
    if(!resize||!swap)throw std::runtime_error("GLM phase backend operations unavailable");
    const size_t alignment=ggml_backend_buft_get_alignment(d.buft);
    auto allocation_size=[&](const auto& group,int slots){
        size_t bytes=0;for(auto* tensor:group.ref)bytes+=((tensor->nb[2]*(slots+1)+alignment-1)/alignment)*alignment;
        return bytes;
    };
    ggml_backend_synchronize(d.backend);
    for(size_t gi=0;gi<p.groups.size();++gi) {
        auto& group=p.groups[gi];auto& lru=group.lru;
        const auto share=cache_budget(p.groups,group,budget,d.host_bytes,
            size_t(double(budget)*group.host_bytes/d.host_bytes),allocation_size,p.n_expert_used);
        const int maximum=int(group.layers.size())*lru.n_expert;
        int low=minimum_cache_routes(p.n_expert_used),high=maximum;
        while(low<high) {
            const int mid=low+(high-low+1)/2;
            if(allocation_size(group,mid)<=share)low=mid;else high=mid-1;
        }
        const int count=low,old=lru.n_slots;
        if(count==old)continue;
        if(count<old) {
            auto selected=lru.key_of;
            std::stable_sort(selected.begin(),selected.end(),[&](int a,int b){
                return a>=0&&(b<0||(lru.frequency&&lru.frequency->score(a)>lru.frequency->score(b)));
            });
            selected.resize(count);
            for(int slot=0;slot<count;++slot)if(selected[slot]>=0) {
                const auto from=std::find(lru.key_of.begin(),lru.key_of.end(),selected[slot])-lru.key_of.begin();
                if(from==slot)continue;
                if(from>=old)throw std::runtime_error("GLM phase compaction lost a key");
                swap(d.backend,group.banks[0]->data,group.banks[0]->nb[2],
                    group.banks[1]->data,group.banks[1]->nb[2],group.banks[2]->data,group.banks[2]->nb[2],slot,int(from));
                std::swap(lru.key_of[slot],lru.key_of[from]);
            }
            ggml_backend_synchronize(d.backend);
        }
        for(size_t role=0;role<3;++role)
            if(!resize(d.phase_banks[3*gi+role].get(),group.banks[role]->nb[2]*(count+1)))
                throw std::runtime_error("GLM phase physical resize failed; no continuation is allowed");
        group.n_slots=count;lru.n_slots=count;lru.key_of.resize(count,-1);
        for(auto* map:lru.slot_map)if(map)std::fill_n(map,lru.n_expert,-1);
        for(int slot=0;slot<count;++slot)if(lru.key_of[slot]>=0) {
            const int key=lru.key_of[slot];lru.slot_map[key/lru.n_expert][key%lru.n_expert]=slot;
        }
        lru.prev.resize(count);lru.next.resize(count);
        for(int slot=0;slot<count;++slot){lru.prev[slot]=slot-1;lru.next[slot]=slot+1<count?slot+1:-1;}
        lru.head=0;lru.tail=count-1;group.native.keys.fill(-1);
        for(auto* bank:group.banks){bank->ne[2]=count+1;bank->nb[3]=bank->nb[2]*bank->ne[2];}
        for(auto& entry:p.bindings)if(p.layers[entry.second.il].ig==int(gi)) {
            auto* view=entry.second.cached;view->ne[2]=count;view->nb[3]=view->nb[2]*view->ne[2];
        }
    }
    d.buf_size=0;for(auto& bank:d.phase_banks)d.buf_size+=ggml_backend_buffer_get_size(bank.get());
}

template<class Impl> void clear_prefill(Impl& p) {
    for(auto& layer:p.layers){layer.prefill_counts.fill(0);layer.prefill_tokens=0;}
}

template<class Impl> void record_prefill(Impl& p,int il,const std::vector<int32_t>& ids) {
    if(ids.size()%p.n_expert_used)throw std::runtime_error("GLM prefill route tuple is incomplete");
    auto& layer=p.layers.at(il);
    for(int id:ids){if(id<0||id>=288)throw std::runtime_error("GLM prefill expert key is invalid");++layer.prefill_counts[id];}
    layer.prefill_tokens+=int(ids.size()/p.n_expert_used);
}

template<class Impl> size_t verify_phase(const Impl& p) {
    size_t bytes=0;
    for(const auto& group:p.groups) {
        const auto& lru=group.lru;
        std::vector<int> resident;
        for(int slot=0;slot<lru.n_slots;++slot) {
            const int key=lru.key_of[slot];if(key<0)continue;
            if(size_t(key/288)>=p.layers.size() || lru.slot_map[key/288][key%288]!=slot)
                throw std::runtime_error("GLM phase slot-to-key link differs");
            resident.push_back(slot);
        }
        for(int il:group.layers)for(int id=0;id<288;++id) {
            const int slot=lru.slot_map[il][id];
            if(slot>=0&&(slot>=lru.n_slots||lru.key_of[slot]!=il*288+id))
                throw std::runtime_error("GLM phase key-to-slot link differs");
        }
        if(resident.empty())continue;
        for(int slot:{resident.front(),resident.back()}) {
            const int key=lru.key_of[slot];
            for(size_t role=0;role<3;++role) {
                const auto* bank=group.banks[role];const auto* source=p.layers[key/288].experts[role];
                std::vector<uint8_t> payload(bank->nb[2]);
                ggml_backend_tensor_get(bank,payload.data(),slot*bank->nb[2],payload.size());
                const auto* original=static_cast<const uint8_t*>(source->data)+(key%288)*source->nb[2];
                if(std::memcmp(payload.data(),original,payload.size())) {
                    const auto bad=std::mismatch(payload.begin(),payload.end(),original).first-payload.begin();
                    throw std::runtime_error("GLM phase payload differs: layer="+std::to_string(key/288)+
                        " expert="+std::to_string(key%288)+" slot="+std::to_string(slot)+" role="+std::to_string(role)+
                        " type="+ggml_type_name(bank->type)+" byte="+std::to_string(bad)+
                        " actual="+std::to_string(payload[bad])+" expected="+std::to_string(original[bad]));
                }
                bytes+=payload.size();
            }
        }
    }
    return bytes;
}

template<class Impl> std::array<uint64_t,4> prefill_coverage(const Impl& p) {
    std::array<uint64_t,4> result{};
    for(const auto& group:p.groups)for(int il:group.layers) {
        const auto& layer=p.layers[il];if(!layer.prefill_tokens)continue;
        const uint64_t tokens=uint64_t(layer.prefill_tokens);
        result[1]=result[0]?std::min(result[1],tokens):tokens;
        result[2]=std::max(result[2],tokens);++result[0];
        for(int e=0;e<288;++e)result[3]+=layer.prefill_counts[e];
    }
    return result;
}

template<class Impl> bool import_prefill(Impl& p) {
    auto* frequency=shared_frequency(p.groups);
    if(!frequency)throw std::runtime_error("GLM phase frequency authority is absent");
    if(!prefill_coverage(p)[0])return false;
    std::vector<uint32_t> state(p.layers.size()*288+1,0);
    frequency->export_state(state.data(),state.size());state.back()=0;
    const char* text=std::getenv("STRATA_GLM53_PREFILL_PRIOR_TOKENS");
    char* end=nullptr;const auto prior=text?std::strtol(text,&end,10):128;
    if(prior<1||prior>16384||(text&&(!*text||*end)))throw std::runtime_error("GLM prefill prior must be1..16384");
    for(const auto& group:p.groups)for(int il:group.layers) {
        const auto& layer=p.layers[il];const int tokens=layer.prefill_tokens;
        // Uncached tiny PP tails have no routing observer. Keep that layer's
        // existing prior; expose actual coverage rather than inventing counts.
        if(!tokens)continue;
        uint64_t total=0;
        for(int e=0;e<288;++e) {
            total+=layer.prefill_counts[e];
            state[il*288+e]=uint32_t((uint64_t(layer.prefill_counts[e])*std::min<long>(prior,tokens)+tokens/2)/tokens);
        }
        if(total!=uint64_t(tokens)*p.n_expert_used)throw std::runtime_error("GLM prefill route occurrence count differs");
    }
    frequency->import_state(state.data(),state.size());return true;
}

template<class Impl,class Upload> size_t seed_phase(Impl& p,Upload upload,uint64_t& removed) {
    import_prefill(p);
    size_t bytes=0;
    for(auto& group:p.groups) {
        auto& lru=group.lru;
        std::vector<std::vector<int>> ranked(group.layers.size());
        std::vector<int> quota(group.layers.size(),0),selected;
        for(size_t il=0;il<ranked.size();++il) {
            auto& r=ranked[il];r.resize(288);std::iota(r.begin(),r.end(),0);
            std::stable_sort(r.begin(),r.end(),[&](int a,int b){return lru.frequency->score(group.layers[il]*288+a)>lru.frequency->score(group.layers[il]*288+b);});
        }
        const int usable=std::max(0,lru.n_slots-p.n_expert_used);
        const int minimum=std::min(p.n_expert_used,usable/int(ranked.size()));
        for(size_t il=0;il<ranked.size();++il)for(int n=0;n<minimum;++n)
            selected.push_back(group.layers[il]*288+ranked[il][quota[il]++]);
        while(int(selected.size())<usable) {
            int best=-1;uint32_t score=0;
            for(size_t il=0;il<ranked.size();++il)if(quota[il]<288) {
                const auto s=lru.frequency->score(group.layers[il]*288+ranked[il][quota[il]]);
                if(best<0||s>score||(s==score&&quota[il]<quota[best])){best=int(il);score=s;}
            }
            if(best<0)break;selected.push_back(group.layers[best]*288+ranked[best][quota[best]++]);
        }
        if(lru.audit)lru.audit->plan();
        const auto previous=lru.key_of;
        std::vector<bool> wanted(p.layers.size()*288,false);for(int key:selected)wanted[key]=true;
        for(int slot=0;slot<lru.n_slots;++slot) {
            const int key=lru.key_of[slot];
            if(key>=0&&!wanted[key]){++removed;lru.slot_map[key/288][key%288]=-1;lru.key_of[slot]=-1;}
        }
        int free=0;
        for(int key:selected)if(lru.slot_map[key/288][key%288]<0) {
            while(free<lru.n_slots&&lru.key_of[free]>=0)++free;
            if(free==lru.n_slots)throw std::runtime_error("GLM phase seed exceeds its owned slots");
            // Upload completion is fenced by the caller before directory publish.
            for(size_t role=0;role<3;++role) {
                const auto* source=p.layers[key/288].experts[role];auto* bank=group.banks[role];
                upload(p.devices[group.id].backend,bank,static_cast<const char*>(source->data)+(key%288)*source->nb[2],free*bank->nb[2],bank->nb[2]);
                bytes+=bank->nb[2];
            }
            if(lru.audit)lru.audit->fill(key,previous[free],lru.frequency->score(key),previous[free]>=0?lru.frequency->score(previous[free]):0);
            lru.key_of[free]=key;lru.slot_map[key/288][key%288]=free;
        }
    }
    return bytes;
}
}
