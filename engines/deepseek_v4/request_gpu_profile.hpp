#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#ifdef STRATA_DEEPSEEK_REQUEST_PROFILER
#include <cuda_profiler_api.h>
#endif
namespace strata::deepseek_v4 {
class RequestGpuProfile {
    bool active=false;
public:
    explicit RequestGpuProfile(bool selected,const char* flag="STRATA_DSV4_PP_ACTIVITY_PROFILE"){
#ifdef STRATA_DEEPSEEK_REQUEST_PROFILER
        const char* value=std::getenv(flag);active=selected&&value&&!std::strcmp(value,"1");
        if(active&&cudaProfilerStart()!=cudaSuccess)throw std::runtime_error("CUDA activity capture start failed");
#else
        (void)selected;(void)flag;
#endif
    }
    ~RequestGpuProfile(){
#ifdef STRATA_DEEPSEEK_REQUEST_PROFILER
        if(active)cudaProfilerStop();
#endif
    }
};
}
