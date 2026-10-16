#pragma once
#include "expert_frequency.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>
namespace strata::deepseek_v4 {
struct BackgroundAdmission { int key=-1,slot=-1,victim=-1; };
using BackgroundAdmissions=std::vector<BackgroundAdmission>;
// Free rows are transaction destinations, absent from both residency maps.
// Existing GPU readers retain every old slot until an idle-boundary commit.
template<class Impl> BackgroundAdmissions plan_background(Impl& p,const uint8_t* eligible) {
    auto& g=p.groups.at(0);auto& lru=g.lru;BackgroundAdmissions plans;
    if(!lru.frequency)return plans;
    std::vector<int> counts(lru.slot_map.size(),0),chosen;
    for(int key:lru.key_of)if(key>=0)++counts[key/lru.n_expert];
    for(int slot=0;slot<lru.n_slots&&plans.size()<6;++slot)if(lru.key_of[slot]<0){
        int key=-1,victim=-1;uint32_t maximum=0,minimum=65536;
        for(int layer:g.layers)for(int expert=0;expert<lru.n_expert;++expert){const int candidate=layer*lru.n_expert+expert;
            if((!eligible||eligible[candidate])&&lru.slot_map[layer][expert]<0&&std::find(chosen.begin(),chosen.end(),candidate)==chosen.end()&&lru.frequency->score(candidate)>maximum){maximum=lru.frequency->score(candidate);key=candidate;}}
        if(key<0)break;
        for(int s=0;s<lru.n_slots;++s){const int old=lru.key_of[s];
            if(old<0||counts[old/lru.n_expert]<=6||std::any_of(plans.begin(),plans.end(),[&](const auto& x){return x.victim==s;}))continue;
            const auto score=lru.frequency->score(old);if(score<minimum){minimum=score;victim=s;}}
        if(victim<0||maximum<=minimum)break;
        plans.push_back({key,slot,victim});chosen.push_back(key);--counts[lru.key_of[victim]/lru.n_expert];++counts[key/lru.n_expert];
    }
    return plans;
}
template<class Impl,class Upload> size_t upload_background(Impl& p,const BackgroundAdmissions& plans,Upload upload) {
    auto& g=p.groups.at(0);size_t bytes=0;
    for(auto item:plans){if(item.key<0||item.slot<0||item.slot>=g.lru.n_slots||g.lru.key_of[item.slot]>=0)throw std::runtime_error("background destination is a live cache slot");
        for(size_t role=0;role<g.banks.size();++role){auto* bank=g.banks[role];const auto* source=p.layers.at(item.key/g.lru.n_expert).experts[role];
            upload(p.devices[g.id].backend,bank,static_cast<const char*>(source->data)+(item.key%g.lru.n_expert)*source->nb[2],item.slot*bank->nb[2],bank->nb[2]);bytes+=bank->nb[2];}}
    return bytes;
}
template<class Impl> void commit_background(Impl& p,const BackgroundAdmissions& plans) {
    auto& lru=p.groups.at(0).lru;
    for(auto item:plans){if(item.slot<0||item.victim<0||lru.key_of.at(item.slot)>=0||lru.key_of.at(item.victim)<0||lru.slot_map.at(item.key/lru.n_expert)[item.key%lru.n_expert]>=0)throw std::runtime_error("background admission lease invalid at idle commit");
        const int old=lru.key_of[item.victim];lru.slot_map[old/lru.n_expert][old%lru.n_expert]=-1;lru.key_of[item.victim]=-1;
        lru.key_of[item.slot]=item.key;lru.slot_map[item.key/lru.n_expert][item.key%lru.n_expert]=item.slot;lru.touch(item.slot);}
}
}
