#pragma once
#include "fp16pv_attention.hpp"
#include <cstdlib>
#include <cstdint>
namespace strata::qwen35::native {
inline bool &fp16pv_request_active(){static thread_local bool active=false;return active;}
inline void configure_fp16pv_request(bool resident){
    // Read optional control once outside Graph; captured shape dispatch is pure.
    const char *flag=std::getenv("STRATA_DENSE_SM120_FP16_PV");
    fp16pv_request_active()=resident&&(!flag||flag[0]!='0');
}
inline bool use_fp16pv(PrefillAttentionShape s){
    // SM120 admission occurs in use_compact_prefill before this callback.
    return fp16pv_request_active()&&!s.graph_start&&s.q8&&s.dim==256&&s.query_heads==24&&s.kv_heads==4&&
           s.tokens>=4096&&s.tokens<=8192;
}
inline std::uint64_t &fp16pv_submissions(){static thread_local std::uint64_t count=0;return count;}
}
