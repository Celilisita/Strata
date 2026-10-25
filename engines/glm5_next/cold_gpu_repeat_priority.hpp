#pragma once
#include "cold_gpu_topology.hpp"
#ifdef __CUDACC__
#define STRATA_REPEAT_HD __host__ __device__
#else
#define STRATA_REPEAT_HD
#endif
namespace strata::glm5_next {
struct ColdGpuRepeatPrioritySelection {
    ColdGpuTopologySelection selected;
    bool retargeted=false;
};
// Mode3 is a K2 replacement of an already-nonzero legacy selection. The
// legacy0 domain invokes the exact existing mode2 algorithm, not a new policy.
// More occurrences is only a routing objective: paired CPU decoding already
// shares weights and may make the complete FFN economics nonpositive.
STRATA_REPEAT_HD inline ColdGpuRepeatPrioritySelection cold_gpu_repeat_priority_selection(
    const ColdGpuLayer& meta,const int32_t* ids,uint32_t missing){
    ColdGpuRepeatPrioritySelection out;
    if(meta.topology_fallback!=3||meta.reserved_rows!=2||(missing&~0xffffu))return out;
    int legacy_keys[2]{-1,-1},legacy_ranks[2]{-1,-1};
    for(int pos=0;pos<2;++pos){int lowest=288;
        for(int rank=pos*8;rank<pos*8+8;++rank){const int key=ids[rank];
            if(!(missing&(1u<<rank))||!cold_gpu_alias_valid(meta,key))continue;
            int occurrences=0;for(int other=0;other<16;++other)occurrences+=ids[other]==key;
            if(occurrences==1&&key<lowest){lowest=key;legacy_keys[pos]=key;legacy_ranks[pos]=rank;}}}
    if(legacy_keys[0]<0||legacy_keys[1]<0){
        out.selected=cold_gpu_topology_selection_mode(meta,ids,missing,2);return out;
    }
    out.selected.count=2;
    for(int i=0;i<2;++i){out.selected.keys[i]=legacy_keys[i];out.selected.ranks[i]=legacy_ranks[i];out.selected.mask|=1u<<legacy_ranks[i];}
    if(out.selected.keys[1]<out.selected.keys[0]){
        const int key=out.selected.keys[0],rank=out.selected.ranks[0];
        out.selected.keys[0]=out.selected.keys[1];out.selected.ranks[0]=out.selected.ranks[1];
        out.selected.keys[1]=key;out.selected.ranks[1]=rank;
    }
    int keys[2]{-1,-1},ranks[2]{-1,-1},counts[2]{};
    for(int rank=0;rank<16;++rank){const int key=ids[rank];
        if(!(missing&(1u<<rank))||!cold_gpu_alias_valid(meta,key))continue;
        bool first=true;for(int other=0;other<rank;++other)if(ids[other]==key)first=false;
        if(!first)continue;
        bool valid=true;uint32_t positions=0;int count=0;
        for(int other=0;other<16;++other)if(ids[other]==key){
            const uint32_t position=1u<<(other/8);
            if(!(missing&(1u<<other))||(positions&position))valid=false;
            positions|=position;++count;
        }
        if(!valid)continue;
        // Original-rank order retains the first key on multiplicity ties.
        if(count>counts[0]){keys[1]=keys[0];ranks[1]=ranks[0];counts[1]=counts[0];keys[0]=key;ranks[0]=rank;counts[0]=count;}
        else if(count>counts[1]){keys[1]=key;ranks[1]=rank;counts[1]=count;}
    }
    const int occurrences=counts[0]+counts[1];int total=0;
    for(uint32_t bits=missing;bits;bits&=bits-1)++total;
    if(keys[1]<0||occurrences<=2||total<=occurrences)return out;
    if(keys[1]<keys[0]){const int key=keys[0],rank=ranks[0];keys[0]=keys[1];ranks[0]=ranks[1];keys[1]=key;ranks[1]=rank;}
    out.selected.mask=0;
    for(int i=0;i<2;++i){out.selected.keys[i]=keys[i];out.selected.ranks[i]=ranks[i];}
    for(int rank=0;rank<16;++rank)if(missing&(1u<<rank))
        for(int i=0;i<2;++i)if(ids[rank]==keys[i])out.selected.mask|=1u<<rank;
    out.retargeted=true;return out;
}
}
#undef STRATA_REPEAT_HD
