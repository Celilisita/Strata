#pragma once
#include <cstdlib>
namespace strata::qwen35::native {
inline bool use_native_operator_fusion(int kind,int tokens){
    static const int forced=[](){const char *p=std::getenv("STRATA_DENSE_NATIVE_OP_FUSION");
        return p&&(p[0]=='0'||p[0]=='1')&&p[1]=='\0'?int(p[0]-'0'):-1;}();
    if(forced>=0)return forced!=0;
    const unsigned mask=kind==0?32u:247u; // MEASURED_MASKS
    const int bucket=tokens<=4?tokens-1:tokens<64?4:tokens<4096?5:tokens<8192?6:7;
    return tokens>0&&(mask&(1u<<bucket));
}
}
