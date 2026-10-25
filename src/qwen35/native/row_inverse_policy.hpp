#pragma once
#include "row_inverse_attention.hpp"
#include "pv_reuse_policy.hpp"
#include <cstdlib>
namespace strata::qwen35::native {
inline bool row_inverse_enabled(){
    static const bool enabled=[](){const char *flag=std::getenv("STRATA_DENSE_SM120_ROW_INVERSE");
        return !flag||flag[0]!='0';}();
    return enabled;
}
inline bool use_row_inverse(PrefillAttentionShape s){return row_inverse_enabled()&&use_pv_reuse(s);}
}
