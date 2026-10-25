#pragma once
#include <cstdlib>
namespace strata::qwen35::native {
inline bool use_conv_gdn(int tokens,int hq,int hv){
    if(tokens<1||hq!=16||hv!=48)return false;
    static const int forced=[](){const char *p=std::getenv("STRATA_DENSE_CONV_GDN_FUSED");
        return p&&(p[0]=='0'||p[0]=='1')&&!p[1]?int(p[0]-'0'):-1;}();
    if(forced>=0)return forced!=0;
    constexpr unsigned mask=255u; // MEASURED_CONV_GDN_MASK
    const int bucket=tokens<=4?tokens-1:tokens<64?4:tokens<4096?5:tokens<8192?6:7;
    return mask&(1u<<bucket);
}
}
