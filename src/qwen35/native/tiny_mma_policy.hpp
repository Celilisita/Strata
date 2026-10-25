#pragma once
#include <cstdlib>
namespace strata::qwen35::native {
inline unsigned tiny_mma_mask(int kind){
    static const int forced=[](){const char *p=std::getenv("STRATA_NVFP4_TINY_MMA");
        return p&&(p[0]=='0'||p[0]=='1')&&!p[1]?int(p[0]-'0'):-1;}();
    if(forced>=0)return forced?15u:0u;
    constexpr unsigned masks[3]={8u,0u,0u}; // MEASURED_TINY_MMA_MASKS
    return masks[kind];
}
inline bool tiny_mma_linear_enabled(int tokens,int columns,int rows){
    const int kind=columns==5120?0:columns==17408?1:-1;
    return rows>=4096&&kind>=0&&tokens>=1&&tokens<=4&&(tiny_mma_mask(kind)&(1u<<(tokens-1)));
}
inline bool tiny_mma_pair_enabled(int tokens,int columns,int rows){
    return rows>=4096&&columns==5120&&tokens>=1&&tokens<=4&&(tiny_mma_mask(2)&(1u<<(tokens-1)));
}
}
