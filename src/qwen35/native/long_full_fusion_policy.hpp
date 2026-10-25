#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35::native {
// Process-cold experiment flag; no environment reads during graph capture/replay.
inline bool long_full_fusion_requested(){
    static const bool enabled=[](){const char*p=std::getenv("STRATA_NVFP4_LONG_FULL_FUSION");
        if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("Long full FFN fusion flag must be0or1");
        return p&&p[0]=='1';}();return enabled;
}
}
