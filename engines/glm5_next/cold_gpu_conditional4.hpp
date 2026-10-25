#pragma once
#include "cold_gpu_protocol.hpp"
#ifdef __CUDACC__
#define STRATA_CONDITIONAL4_HD __host__ __device__
#else
#define STRATA_CONDITIONAL4_HD
#endif

namespace strata::glm5_next {
// This uses the original, pre-offload missing mask. Main39 geometry/type and
// reservation/epoch remain the existing owner's responsibility.
STRATA_CONDITIONAL4_HD inline bool cold_gpu_conditional4_shape(const int32_t* ids,uint32_t missing) {
    if(!ids||(missing&~0xffffu))return false;
    unsigned count=0,per_position[2]{};
    for(unsigned rank=0;rank<16;++rank){
        if(ids[rank]<0||ids[rank]>=288)return false;
        for(unsigned previous=0;previous<rank;++previous)if(ids[previous]==ids[rank])return false;
        if(missing&(1u<<rank)){++count;++per_position[rank/8];}
    }
    return count==12&&per_position[0]>=2&&per_position[1]>=2;
}
// Mode2 is fixture-only force2 under the same four-row reservation. Neither
// metadata selection nor a shape match manufactures an alias or a lease.
STRATA_CONDITIONAL4_HD inline bool cold_gpu_allow_four(const ColdGpuLayer& meta,const int32_t* ids,uint32_t missing) {
    if(meta.conditional4_policy==0)return true;
    return meta.conditional4_policy==1&&cold_gpu_conditional4_shape(ids,missing);
}
}
#undef STRATA_CONDITIONAL4_HD
