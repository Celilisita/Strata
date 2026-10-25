#pragma once
#include "q6_warp_logical.hpp"
#include <cstdlib>
namespace strata::qwen35::native {
inline bool &q6_warp_admission(){static thread_local bool allowed=false;return allowed;}
inline void configure_q6_warp_head(){
    // Called by the service profile before prefill/graph capture on its thread.
    const char *flag=std::getenv("STRATA_DENSE_Q6_WARP_HEAD");
    if(flag&&flag[0]=='0'){q6_warp_admission()=false;return;}
    int device=0;check_cuda(cudaGetDevice(&device),"Q6 warp head device");const auto caps=capabilities(device);
    q6_warp_admission()=caps.major==12&&caps.minor==0;
}
inline bool use_q6_warp_head(Matrix w,int tokens){
    constexpr unsigned mask=12;
    return q6_warp_admission()&&w.quant==Quant::Q6K&&w.layout==BlockLayout::Gguf&&w.rows==248320&&w.columns==5120&&
        tokens>=1&&tokens<=4&&(mask&(1u<<(tokens-1)));
}
}
