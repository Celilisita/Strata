#pragma once
#include "route_trace_protocol.hpp"
namespace strata::glm5_next {
__device__ inline void capture_routes(RouteTraceHeader* trace,int layer,const int32_t* ids,
                                     const int32_t* slots,int count) {
    if(!trace||!trace->enabled)return;
    const uint32_t index=atomicAdd(&trace->count,1u);
    if(index>=route_trace_capacity){atomicAdd(&trace->overflow,1u);return;}
    auto* rows=reinterpret_cast<RouteTraceRecord*>(trace+1);
    auto& row=rows[index];row.layer=uint32_t(layer);row.count=uint32_t(count);
    row.missing=0;row.reserved=0;
    for(int i=0;i<16;++i){row.ids[i]=i<count?uint16_t(ids[i]):0;
        if(i<count&&slots[i]<0)row.missing|=1u<<i;}
}
}
