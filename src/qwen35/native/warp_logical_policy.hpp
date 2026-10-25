#pragma once
#include "strata/qwen35/native/ffn.hpp"
#include <cstdlib>
namespace strata::qwen35::native {
inline bool warp_logical_supported(){
    const char *flag=std::getenv("STRATA_NVFP4_WARP_LOGICAL");
    if(flag&&flag[0]=='0')return false;
    int device=0;check_cuda(cudaGetDevice(&device),"logical warp policy device");
    const auto caps=capabilities(device);return caps.major==12&&caps.minor==0;
}
inline bool warp_logical_linear_enabled(int tokens,int columns,int rows){
    constexpr unsigned mask=4u;
    return columns==17408&&rows==5120&&tokens>=1&&tokens<=3&&(mask&(1u<<(tokens-1)));
}
inline bool warp_logical_pair_enabled(int tokens,int columns,int rows){
    constexpr unsigned mask=6u;
    return columns==5120&&rows==17408&&tokens>=1&&tokens<=3&&(mask&(1u<<(tokens-1)));
}
}
