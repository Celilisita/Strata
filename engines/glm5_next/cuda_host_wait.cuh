#pragma once
// Host scheduling only. No kernels, graph math, stream ordering or cache data.
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
inline bool cpu_async_enqueue_requested() {
    const char* flag=std::getenv("STRATA_GLM53_HYBRID_ENQUEUE");
    if(flag&&std::strcmp(flag,"0")&&std::strcmp(flag,"1"))
        throw std::runtime_error("GLM CPU async enqueue must be0 or1");
    return flag&&!std::strcmp(flag,"1");
}
#ifdef __CUDACC__
inline void configure_cuda_host_wait(int device) {
    const char* flag=std::getenv("STRATA_GLM53_CUDA_BLOCKING_WAIT");
    if(flag&&std::strcmp(flag,"0")&&std::strcmp(flag,"1"))
        throw std::runtime_error("GLM CUDA blocking wait must be0 or1");
    if(!flag||std::strcmp(flag,"1"))return;
    const char* ccd=std::getenv("STRATA_GLM53_CPU_SINGLE_CCD");
    if(!ccd||std::strcmp(ccd,"1"))throw std::runtime_error("GLM blocking wait requires the one-CCD profile");
    CUDA_CHECK(cudaSetDevice(device));
    unsigned previous=0;CUDA_CHECK(cudaGetDeviceFlags(&previous));
    const unsigned desired=(previous&~unsigned(cudaDeviceScheduleMask))|cudaDeviceScheduleBlockingSync;
    // Runtime reports MapHost implicitly; CUDA13 excludes it from accepted
    // cudaSetDeviceFlags arguments while preserving mapped-host capability.
    if(previous!=desired)CUDA_CHECK(cudaSetDeviceFlags(desired&~unsigned(cudaDeviceMapHost)));
    unsigned actual=0;CUDA_CHECK(cudaGetDeviceFlags(&actual));
    if(actual!=desired)throw std::runtime_error("CUDA host wait scheduling flag was not applied");
}
#endif
}
