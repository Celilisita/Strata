#pragma once
#include "cold_gpu_protocol.hpp"

#ifdef __CUDACC__
#define STRATA_COLD_TOPOLOGY_HD __host__ __device__
#else
#define STRATA_COLD_TOPOLOGY_HD
#endif

namespace strata::glm5_next {
struct ColdGpuTopologySelection {
    uint32_t mask=0,count=0;
    int keys[2]{-1,-1},ranks[2]{-1,-1};
};

// Called only when the old balanced, globally-unique selection has mask0.
// First-seen original rank chooses keys; sorting changes payload row order only.
STRATA_COLD_TOPOLOGY_HD inline ColdGpuTopologySelection cold_gpu_topology_selection_mode(
    const ColdGpuLayer& meta,const int32_t* ids,uint32_t missing,uint32_t topology_mode) {
    ColdGpuTopologySelection out;
    if((topology_mode!=1&&topology_mode!=2)||meta.reserved_rows!=2||(missing&~0xffffu))return out;
    for(int rank=0;rank<16&&out.count<2;++rank){
        const int key=ids[rank];
        if(!(missing&(1u<<rank))||!cold_gpu_alias_valid(meta,key))continue;
        if(out.count&&key==out.keys[0])continue;
        bool valid=true;uint32_t positions=0;
        for(int other=0;other<16;++other)if(ids[other]==key){
            const uint32_t position=1u<<(other/8);
            if(!(missing&(1u<<other))||(positions&position))valid=false;
            positions|=position;
        }
        if(!valid)continue;
        out.keys[out.count]=key;out.ranks[out.count]=rank;++out.count;
    }
    if(out.count==2&&out.keys[1]<out.keys[0]){
        const int key=out.keys[0],rank=out.ranks[0];
        out.keys[0]=out.keys[1];out.ranks[0]=out.ranks[1];
        out.keys[1]=key;out.ranks[1]=rank;
    }
    for(int rank=0;rank<16;++rank)if(missing&(1u<<rank))
        for(uint32_t i=0;i<out.count;++i)if(ids[rank]==out.keys[i])out.mask|=1u<<rank;
    // Mode2 prices only the measured shared-two-key domain and keeps CPU work.
    // The legacy nonzero selector returns before this fallback is consulted.
    if(topology_mode==2){
        uint32_t occurrences=0,total=0;
        for(uint32_t bits=out.mask;bits;bits&=bits-1)++occurrences;
        for(uint32_t bits=missing;bits;bits&=bits-1)++total;
        if(out.count!=2||occurrences<3||total<=occurrences)return {};
    }
    return out;
}
STRATA_COLD_TOPOLOGY_HD inline ColdGpuTopologySelection cold_gpu_topology_selection(
    const ColdGpuLayer& meta,const int32_t* ids,uint32_t missing){
    return cold_gpu_topology_selection_mode(meta,ids,missing,meta.topology_fallback);
}
STRATA_COLD_TOPOLOGY_HD inline uint32_t cold_gpu_occurrences(uint32_t mask){
    uint32_t count=0;for(;mask;mask&=mask-1)++count;return count;
}
}
#undef STRATA_COLD_TOPOLOGY_HD
