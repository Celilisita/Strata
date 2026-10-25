#pragma once
#include "gutail_encode.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cstdlib>
#include <cstdint>
#include <string>
namespace strata::qwen35::native {
// PREPARE-only flags and eager/capture diagnostics; Graph replay retains no host state.
inline std::uint64_t &fp4_gu_tail_host_submissions(){static thread_local std::uint64_t count=0;return count;}
inline bool fp4_gu_tail_prepare(bool eligible){
    const char *raw=std::getenv("STRATA_NVFP4_GU_TAIL");const std::string mode=raw?raw:"auto";
    if(mode!="auto"&&mode!="0"&&mode!="1")throw std::invalid_argument("GU tail mode must be auto,0,1");
    fp4_gu_tail_host_submissions()=0;
    if(mode=="0"||!eligible)return false;
    int device=0;check_cuda(cudaGetDevice(&device),"GU tail PREPARE device");const auto caps=capabilities(device);
    if(caps.major!=12||caps.minor!=0)return false;
    gutail_configure_kernel();const auto resources=gutail_resources();
    if(resources.blocks<1||resources.local_bytes)throw std::runtime_error("GU tail resource admission");
    return true;
}
inline void fp4_gu_tail_project(PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,int tokens,bool half,float scale,cudaStream_t stream){
    if(tokens==5424){gutail_fused(g,u,x,out,tokens,half,scale,stream);++fp4_gu_tail_host_submissions();}
    else fp4_gu_fused(g,u,x,out,tokens,half,scale,stream);
}
}
