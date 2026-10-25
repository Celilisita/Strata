#pragma once
#include <utility>
#include "cold_gpu_protocol.hpp"
#include "cold_gpu_topology.hpp"
#include "cold_gpu_repeat_priority.hpp"
#include "cold_gpu_conditional4.hpp"
#include <algorithm>
namespace strata::glm5_next {
struct ColdGpuSelection {
    uint32_t mask=0,count=0,topology=0;
    int keys[4]{-1,-1,-1,-1},ranks[4]{-1,-1,-1,-1};
};
// Pure host mirror; retain the original lowest-one scan for capacity2.
inline ColdGpuSelection cold_gpu_host_selection(const ColdGpuLayer& meta,const int32_t* ids,uint32_t missing) {
    if(meta.topology_fallback==3&&meta.reserved_rows==2){
        const auto next=cold_gpu_repeat_priority_selection(meta,ids,missing);ColdGpuSelection selected;
        selected.mask=next.selected.mask;selected.count=next.selected.count;
        selected.topology=next.retargeted?2u:(cold_gpu_occurrences(selected.mask)>2?1u:0u);
        for(uint32_t i=0;i<selected.count;++i){selected.keys[i]=next.selected.keys[i];selected.ranks[i]=next.selected.ranks[i];}
        return selected;
    }
    ColdGpuSelection out;
    if(meta.reserved_rows!=2&&meta.reserved_rows!=4)return out;
    for(int pos=0;pos<2;++pos){int lowest=288;
        for(int rank=pos*8;rank<pos*8+8;++rank){const int key=ids[rank];
            if(!(missing&(1u<<rank))||!cold_gpu_alias_valid(meta,key))continue;
            int occurrences=0;for(int other=0;other<16;++other)occurrences+=ids[other]==key;
            if(occurrences==1&&key<lowest){lowest=key;out.keys[pos]=key;out.ranks[pos]=rank;}}}
    if(out.keys[0]<0||out.keys[1]<0){
        if(meta.topology_fallback){
            const auto next=cold_gpu_topology_selection(meta,ids,missing);
            out={};out.mask=next.mask;out.count=next.count;out.topology=next.mask?1u:0u;
            for(uint32_t i=0;i<next.count;++i){out.keys[i]=next.keys[i];out.ranks[i]=next.ranks[i];}
        }
        return out;
    }
    out.count=2;
    if(meta.reserved_rows==4&&cold_gpu_allow_four(meta,ids,missing)){
        for(int pos=0;pos<2;++pos){int lowest=288;
            for(int rank=pos*8;rank<pos*8+8;++rank){const int key=ids[rank];
                if(!(missing&(1u<<rank))||key==out.keys[pos]||!cold_gpu_alias_valid(meta,key))continue;
                int occurrences=0;for(int other=0;other<16;++other)occurrences+=ids[other]==key;
                if(occurrences==1&&key<lowest){lowest=key;out.keys[pos+2]=key;out.ranks[pos+2]=rank;}}}
        if(out.keys[2]>=0&&out.keys[3]>=0)out.count=4;
        else{out.keys[2]=out.keys[3]=out.ranks[2]=out.ranks[3]=-1;}
    }
    for(uint32_t i=1;i<out.count;++i)for(uint32_t j=i;j&&out.keys[j]<out.keys[j-1];--j){
        std::swap(out.keys[j],out.keys[j-1]);std::swap(out.ranks[j],out.ranks[j-1]);}
    for(uint32_t i=0;i<out.count;++i)out.mask|=1u<<out.ranks[i];
    return out;
}
}
