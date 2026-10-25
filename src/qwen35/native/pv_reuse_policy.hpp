#pragma once
#include "pv_reuse_attention.hpp"
#include <cstdlib>
namespace strata::qwen35::native {
inline bool pv_reuse_enabled(){
    static const bool enabled=[](){const char *flag=std::getenv("STRATA_DENSE_SM120_PV_REUSE");
        return !flag||flag[0]!='0';}();
    return enabled;
}
inline bool use_pv_reuse(PrefillAttentionShape s){
    // These two shape classes have positive measured complete-operator costs.
    // SM120 admission is supplied by use_compact_prefill before this callback.
    return pv_reuse_enabled()&&!s.graph_start&&s.q8&&s.dim==256&&s.query_heads==24&&s.kv_heads==4&&
        ((s.query_start==0&&s.tokens==8192&&s.valid==8192)||
         (s.query_start==24576&&s.tokens==5424&&s.valid==30000));
}
}
