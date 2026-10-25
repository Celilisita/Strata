#pragma once
#include "attention_rows64.hpp"
namespace strata::qwen35moe {
inline bool attention_rows64_qpad8_flag(const char* value) {
    if (!value) return false;
    if ((value[0]!='0' && value[0]!='1') || value[1]!='\0')
        throw std::invalid_argument("STRATA_QWEN35_ATTN_ROWS64_QPAD8 must be0 or1");
    return value[0]=='1';
}
bool attention_rows64_qpad8_selected(int,int,int);
AttentionRows64Resources attention_rows64_qpad8_resources();
bool attention_rows64_qpad8_prefill(const float*,uint16_t*,uint16_t*,int,int,int,float*,void*);
// Host submissions include capture; graph replay does not increment this counter.
unsigned long long attention_rows64_qpad8_submissions();
}
