#pragma once
#include "expert_frequency.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace strata::deepseek_v4 {
struct BackgroundAdmission { int key=-1,slot=-1,victim=-1,group=0; };
using BackgroundAdmissions=std::vector<BackgroundAdmission>;
// Free rows are transaction destinations, absent from both residency maps.
// Existing GPU readers retain every old slot until an idle-boundary commit.
template<class Impl> BackgroundAdmissions plan_background(Impl& p,const uint8_t* eligible) {
    BackgroundAdmissions plans;
    for(size_t gi=0;gi<p.groups.size();++gi) {
    auto& g=p.groups[gi];auto& lru=g.lru;
    if(!lru.frequency)continue;
    const size_t begin=plans.size();
    const int floor= lru.n_expert==288 ? std::min(p.n_expert_used,
        std::max(0,lru.n_slots-p.n_expert_used)/int(g.layers.size())) : 6;
    std::vector<int> counts(lru.slot_map.size(),0),chosen;
    for(int key:lru.key_of)if(key>=0)++counts[key/lru.n_expert];
    for(int slot=0;slot<lru.n_slots&&plans.size()-begin<6;++slot)if(lru.key_of[slot]<0){
        int key=-1,victim=-1;uint32_t maximum=0,minimum=65536;
        for(int layer:g.layers)for(int expert=0;expert<lru.n_expert;++expert){const int candidate=layer*lru.n_expert+expert;
            if((!eligible||eligible[candidate])&&lru.slot_map[layer][expert]<0&&std::find(chosen.begin(),chosen.end(),candidate)==chosen.end()&&lru.frequency->score(candidate)>maximum){maximum=lru.frequency->score(candidate);key=candidate;}}
        if(key<0)break;
        for(int s=0;s<lru.n_slots;++s){const int old=lru.key_of[s];
            if(old<0||counts[old/lru.n_expert]<=floor||std::any_of(plans.begin()+begin,plans.end(),[&](const auto& x){return x.victim==s;}))continue;
            const auto score=lru.frequency->score(old);if(score<minimum){minimum=score;victim=s;}}
        if(victim<0||maximum<=minimum)break;
        plans.push_back({key,slot,victim,int(gi)});chosen.push_back(key);--counts[lru.key_of[victim]/lru.n_expert];++counts[key/lru.n_expert];
    }
    }
    return plans;
}
template<class Impl,class Upload> size_t upload_background(Impl& p,const BackgroundAdmissions& plans,Upload upload) {
    size_t bytes=0;
    for(auto item:plans){auto& g=p.groups.at(item.group);
        if(item.key<0||p.layers.at(item.key/g.lru.n_expert).ig!=item.group||item.slot<0||item.slot>=g.lru.n_slots||g.lru.key_of[item.slot]>=0)throw std::runtime_error("background destination is a live cache slot");
        for(size_t role=0;role<g.banks.size();++role){auto* bank=g.banks[role];const auto* source=p.layers.at(item.key/g.lru.n_expert).experts[role];
            upload(p.devices[g.id].backend,bank,static_cast<const char*>(source->data)+(item.key%g.lru.n_expert)*source->nb[2],item.slot*bank->nb[2],bank->nb[2]);bytes+=bank->nb[2];}}
    return bytes;
}
template<class Impl> void commit_background(Impl& p,const BackgroundAdmissions& plans) {
    for(auto item:plans){const auto& lru=p.groups.at(item.group).lru;
        if(item.key<0||p.layers.at(item.key/lru.n_expert).ig!=item.group||item.slot<0||item.victim<0||lru.key_of.at(item.slot)>=0||lru.key_of.at(item.victim)<0||lru.slot_map.at(item.key/lru.n_expert)[item.key%lru.n_expert]>=0)throw std::runtime_error("background admission lease invalid at idle commit");}
    for(auto item:plans){auto& lru=p.groups.at(item.group).lru;
        const int old=lru.key_of[item.victim];lru.slot_map[old/lru.n_expert][old%lru.n_expert]=-1;lru.key_of[item.victim]=-1;
        lru.key_of[item.slot]=item.key;lru.slot_map[item.key/lru.n_expert][item.key%lru.n_expert]=item.slot;lru.touch(item.slot);}
}

// Diagnostic only, after the payload event and before directory publication.
template<class Impl> size_t verify_background(const Impl& p,const BackgroundAdmissions& plans) {
    size_t bytes=0;
    for(auto item:plans) {
        const auto& g=p.groups.at(item.group);const auto& lru=g.lru;
        if(item.key<0||p.layers.at(item.key/lru.n_expert).ig!=item.group||lru.key_of.at(item.slot)>=0)
            throw std::runtime_error("background verification lost its free-row lease");
        for(size_t role=0;role<g.banks.size();++role) {
            const auto* bank=g.banks[role];const auto* source=p.layers.at(item.key/lru.n_expert).experts[role];
            if(bank->nb[2]!=source->nb[2])throw std::runtime_error("background payload stride differs");
            std::vector<uint8_t> actual(bank->nb[2]);
            ggml_backend_tensor_get(bank,actual.data(),size_t(item.slot)*bank->nb[2],actual.size());
            if(std::memcmp(actual.data(),static_cast<const char*>(source->data)+(item.key%lru.n_expert)*source->nb[2],actual.size()))
                throw std::runtime_error("background payload differs from original GGUF");
            bytes+=actual.size();
        }
    }
    return bytes;
}
}
