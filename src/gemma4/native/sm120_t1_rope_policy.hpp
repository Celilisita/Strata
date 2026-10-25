#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::gemma4::native {
// This private host policy is instantiated only in sm120_model_forward.cpp.
// The first forward fixes the process policy; Graph replay reads device position.
inline bool t1_rope_table_requested(){
    static const bool enabled=[](){
        const char* value=std::getenv("STRATA_GEMMA_T1_ROPE_TABLE");
        if(!value||std::strcmp(value,"0")==0)return false;
        if(std::strcmp(value,"1")==0)return true;
        throw std::invalid_argument("Gemma T1 RoPE table policy requires0 or1");
    }();
    return enabled;
}
}
