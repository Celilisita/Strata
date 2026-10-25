#pragma once
#include "fp4_tiny_fused.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cstdlib>
#include <cstdint>
#include <string>
namespace strata::qwen35::native {
PackedFp4Resources fp4_gu8_resources();
void fp4_tiny_gu8_pair(PackedFp4,PackedFp4,CompensatedFp4Input,float,int,cudaStream_t);
inline std::uint64_t &fp4_gu8_host_submissions(){static thread_local std::uint64_t count=0;return count;}
inline bool fp4_gu8_prepare(bool eligible){
    const char *raw=std::getenv("STRATA_NVFP4_GU8");const std::string mode=raw?raw:"auto";
    if(mode!="auto"&&mode!="0"&&mode!="1")throw std::invalid_argument("GU8 mode must be auto,0,1");
    fp4_gu8_host_submissions()=0;if(mode=="0"||!eligible)return false;
    int device=0;check_cuda(cudaGetDevice(&device),"GU8 PREPARE device");const auto caps=capabilities(device);
    if(caps.major!=12||caps.minor!=0)return false;
    const auto resources=fp4_gu8_resources();
    if(resources.blocks<1||resources.local_bytes)throw std::runtime_error("GU8 resource admission");
    return true;
}
// Runtime owns the four encoded arenas; this selector changes only T3 GU.
inline void fp4_gu8_project(bool enabled,PackedFp4 g,PackedFp4 u,CompensatedFp4Input x,float scale,int tokens,cudaStream_t stream){
    if(enabled&&tokens==3){fp4_tiny_gu8_pair(g,u,x,scale,tokens,stream);++fp4_gu8_host_submissions();}
    else fp4_tiny_fused_pair(g,u,x,scale,tokens,stream);
}
}
