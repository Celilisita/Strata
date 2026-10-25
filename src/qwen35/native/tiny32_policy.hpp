#pragma once
#include <cstdlib>
namespace strata::qwen35::native {
inline unsigned tiny32_mask(int kind){
    static const int forced=[](){const char *p=std::getenv("STRATA_NVFP4_TINY32");
        return p&&(p[0]=='0'||p[0]=='1')&&!p[1]?int(p[0]-'0'):-1;}();
    if(forced>=0)return forced?15u:0u;
    constexpr unsigned masks[3]={15u,0u,8u}; // MEASURED_TINY32_MASKS
    return masks[kind];
}
inline bool tiny32_linear_enabled(int tokens,int columns){
    const int kind=columns==5120?0:columns==17408?1:-1;
    return kind>=0&&tokens>=1&&tokens<=4&&(tiny32_mask(kind)&(1u<<(tokens-1)));
}
inline bool tiny32_pair_enabled(int tokens,int columns){
    return columns==5120&&tokens>=1&&tokens<=4&&(tiny32_mask(2)&(1u<<(tokens-1)));
}
}
