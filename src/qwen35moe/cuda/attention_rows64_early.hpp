#pragma once
#include "attention_rows64.hpp"
namespace strata::qwen35moe {
inline bool attention_rows64_early_flag(const char* value) {
    if (!value) return false;
    if ((value[0]!='0' && value[0]!='1') || value[1]!='\0')
        throw std::invalid_argument("STRATA_QWEN35_ATTN_ROWS64_EARLY must be0 or1");
    return value[0]=='1';
}
inline bool attention_rows64_early_candidate(int first,int count,int capacity) noexcept {
    return attention_tc_fits(first,count,capacity) && capacity<=40960 &&
        ((first==256 && count==1792) || (first==2048 && count==2048) ||
         (first==4096 && count==2048));
}
bool attention_rows64_early_selected(int,int,int);
AttentionRows64Resources attention_rows64_early_resources();
bool attention_rows64_early_prefill(const float*,uint16_t*,uint16_t*,int,int,int,float*,void*);
// Successful host submissions include capture; graph replay does not increment.
unsigned long long attention_rows64_early_submissions();
}
