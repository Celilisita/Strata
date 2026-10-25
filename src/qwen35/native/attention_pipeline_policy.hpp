#pragma once
#include <cstdlib>
namespace strata::qwen35::native {
// The task driver fills these constants from positive measured operator gains.
inline int attention_pipeline_mode(int query_start){
    static const int forced=[](){
        const char *flag=std::getenv("STRATA_DENSE_SM120_ATTENTION_PIPELINE");
        return flag&&flag[0]>='0'&&flag[0]<='2'&&flag[1]=='\0'?int(flag[0]-'0'):-1;
    }();
    if(forced>=0)return forced;
    return query_start==0?2:2; // MEASURED_POLICY
}
}
