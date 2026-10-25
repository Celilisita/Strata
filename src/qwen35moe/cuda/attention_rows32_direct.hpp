#pragma once
#include "strata/qwen35moe/attention_tc.hpp"
#include <stdexcept>
namespace strata::qwen35moe {
struct AttentionRows32DirectResources {
    int rows,threads,registers,active_blocks,binary_version,reason_mask;
    size_t static_shared,dynamic_shared,local_bytes,optin_limit;
};
inline bool attention_rows32_direct_flag(const char* flag) {
    if (!flag) return false;
    if ((flag[0]!='0' && flag[0]!='1') || flag[1]!='\0')
        throw std::invalid_argument("ATTN_QUERY_ROWS32_DIRECT must be0 or1");
    return flag[0]=='1';
}
inline bool attention_rows32_direct_candidate(int first,int count,int capacity) noexcept {
    return attention_tc_fits(first,count,capacity) && count>=1024 &&
        capacity<=40960;
}
// Caller retains the original Reuse/Register gate; fixed D256/16Q/2KV heads.
// Explicit request shares the exact complete production fallback, for the linked qualifier.
void attention_tc_prefill_rows32_direct_request(const float*,uint16_t*,uint16_t*,int,int,int,float*,void*,bool,bool,bool);
bool attention_rows32_direct_selected(int,int,int);
AttentionRows32DirectResources attention_rows32_direct_resources();
// False before mutation on geometry/resource fallback. CUDA errors throw.
bool attention_rows32_direct_prefill(const float*,uint16_t*,uint16_t*,int,int,int,float*,void*);
// Successful host submissions, including capture; graph replay is not counted.
unsigned long long attention_rows32_direct_submissions();
}
