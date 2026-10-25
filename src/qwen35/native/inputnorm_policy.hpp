#pragma once
#include "inputnorm.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
inline bool &inputnorm_request_active(){static thread_local bool active=false;return active;}
// Host eager/capture producer count; Graph replay retains no host descriptor.
inline std::uint64_t &inputnorm_host_submissions(){static thread_local std::uint64_t count=0;return count;}
inline void configure_inputnorm_request(bool admitted){
    inputnorm_request_active()=false;inputnorm_host_submissions()=0;
    const char *raw=std::getenv("STRATA_NVFP4_INPUT_NORM");const std::string mode=raw?raw:"auto";
    if(mode!="auto"&&mode!="0"&&mode!="1")throw std::invalid_argument("input RMS mode must be auto,0,1");
    if(mode=="0"||!admitted)return;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
    int device=0;check_cuda(cudaGetDevice(&device),"input RMS PREPARE device");const auto caps=capabilities(device);
    if(caps.major!=12||caps.minor!=0)return;
    const auto resources=inputnorm_resources();
    if(resources.blocks<1||resources.local_bytes)throw std::runtime_error("input RMS resource admission");
    inputnorm_request_active()=true;
#endif
}
}
