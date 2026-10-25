#pragma once
#include <cstdlib>
namespace strata::qwen35::native {
inline bool use_vector_swiglu(int tokens,int columns){
    static const int forced=[](){const char *p=std::getenv("STRATA_NVFP4_VECTOR_SWIGLU");
        return p&&(p[0]=='0'||p[0]=='1')&&!p[1]?int(p[0]-'0'):-1;}();
    if(forced>=0)return forced!=0;
    const unsigned mask=7u; // MEASURED_VECTOR_MASK
    const int bucket=tokens<4096?0:tokens<8192?1:2;
    return tokens>=128&&columns==17408&&(mask&(1u<<bucket));
}
}
