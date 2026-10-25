#pragma once
#include "fp4_gu_merge.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35::native {
struct HalfMergeStats {bool requested=false,admitted=false;uint64_t host_calls=0;int prompt=0;};
inline HalfMergeStats &halfmerge_request_state(){static thread_local HalfMergeStats value;return value;}
inline HalfMergeStats halfmerge_stats(){return halfmerge_request_state();}
inline bool halfmerge_requested(){
    static const bool enabled=[](){const char *p=std::getenv("STRATA_NVFP4_LONG_HALF_MERGE");
        if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("Long Half merged GU must be0or1");
        return p&&p[0]=='1';}();return enabled;
}
inline bool halfmerge_tma_prepare(bool &long_request,bool allowed,int prompt,int capacity,const char *tma){
    const char *merged=std::getenv("STRATA_NVFP4_GU_MERGED");
    long_request=halfmerge_requested()&&allowed&&prompt>8192&&capacity>=8192&&(!tma||tma[0]!='0')&&(!merged||merged[0]!='0');
    halfmerge_request_state()={halfmerge_requested(),long_request,0,prompt};
    return allowed&&(prompt==8192||long_request)&&capacity>=8192&&(!tma||tma[0]!='0');
}
inline void halfmerge_dispatch(const GuMergePlan &p,PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input y,int t,bool h,float scale,cudaStream_t stream,bool long_request){
    if(long_request)++halfmerge_request_state().host_calls;
    fp4_gu_merge(p,g,u,x,y,t,h,scale,stream);
}
}
