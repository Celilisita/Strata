#pragma once
#include "half_tail.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35::native {
struct HalfTailStats {bool requested=false,admitted=false;int prompt=0,last_tokens=0,last_capacity=0;std::uint64_t host_calls=0;HalfTailResources resource{};};
inline HalfTailStats& half_tail_stats(){static thread_local HalfTailStats value;return value;}
inline bool half_tail_requested(){static const bool enabled=[](){const char*p=std::getenv("STRATA_NVFP4_HALF_GU_TAIL");
    if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("Half GU tail flag must be0or1");return p&&p[0]=='1';}();return enabled;}
inline void half_tail_prepare_request(bool allowed,int prompt){auto&v=half_tail_stats();v={};v.requested=half_tail_requested();v.prompt=prompt;
#if CUDART_VERSION >= 12080
    if(!v.requested||!allowed||prompt<=8192||prompt>32768||prompt%8192!=5424)return;
    int device=0;check_cuda(cudaGetDevice(&device),"Half GU tail device");const auto caps=capabilities(device);if(caps.major!=12||caps.minor!=0)return;
    half_tail_configure();v.resource=half_tail_resources();const auto&r=v.resource;
    if(r.blocks<1||r.local_bytes||r.binary_version!=120||r.kernel_count!=2)throw std::runtime_error("Half GU tail requires qualified SM120 zero-local resources");v.admitted=true;
#else
    (void)allowed;(void)prompt;
#endif
}
inline bool half_tail_admit(bool merged,int tokens,bool diagnostic){return half_tail_stats().admitted&&merged&&tokens==5424&&!diagnostic;}
inline void half_tail_submit(const GuMergePlan&p,PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,int t,bool h,float scale,cudaStream_t s){
#if CUDART_VERSION >= 12080
    half_tail(p,g,u,x,out,t,h,scale,s);auto&v=half_tail_stats();++v.host_calls;v.last_tokens=t;v.last_capacity=p.capacity;
#else
    (void)p;(void)g;(void)u;(void)x;(void)out;(void)t;(void)h;(void)scale;(void)s;throw std::logic_error("Half tail unavailable CUDA toolkit");
#endif
}
}
