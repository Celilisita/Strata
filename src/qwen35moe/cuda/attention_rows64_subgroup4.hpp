#pragma once
#include "attention_rows64.hpp"
namespace strata::qwen35moe {
inline bool attention_rows64_subgroup4_flag(const char* flag) {
    if(!flag)return false;
    if((flag[0]!='0'&&flag[0]!='1')||flag[1]!='\0')
        throw std::invalid_argument("ATTN_ROWS64_SUBGROUP4 must be0 or1");
    return flag[0]=='1';
}
bool attention_rows64_subgroup4_selected(int,int,int);
AttentionRows64Resources attention_rows64_subgroup4_resources();
// Explicit primitive: no environment mutation is needed for same-process A/B.
// False before writes for ineligible shape/resources; actual CUDA errors throw.
bool attention_rows64_subgroup4_prefill(const float*,uint16_t*,uint16_t*,int,int,int,float*,void*);
unsigned long long attention_rows64_subgroup4_submissions();
}
