#pragma once
#include "strata/qwen35/native/ffn.hpp"
#include "strata/qwen35/native/weights.hpp"
#include <cstdlib>
#include <string>
namespace strata::qwen35::native {
inline bool resident_kv_requested(int required,int resident){
    // Reserve three extra positions for the largest supported draft lookahead.
    // This selects the whole fresh request, never just its current PP prefix.
    if(required<1||required>resident-3)return false;
    const char *flag=std::getenv("STRATA_DENSE_RESIDENT_KV");
    if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("resident KV mode");
    if(flag&&flag[0]=='0')return false;
    int device=0;check_cuda(cudaGetDevice(&device),"resident KV provider device");
    const auto caps=capabilities(device);return caps.major==12&&caps.minor==0;
}
}
