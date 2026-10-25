#pragma once
#include "attention_rows64_early.hpp"
#include "attention_rows64_qpad8.hpp"
namespace strata::qwen35moe {
inline bool attention_rows64_early_qpad8_candidate(bool early,bool qpad,int first,int count,int capacity) noexcept {
    return early&&qpad&&attention_rows64_early_candidate(first,count,capacity);
}
// Existing startup EARLY1+QPAD8=1 conjunction only. No new flag or runtime toggle.
bool attention_rows64_early_qpad8_selected(int,int,int);
AttentionRows64Resources attention_rows64_early_qpad8_resources();
bool attention_rows64_early_qpad8_prefill(const float*,uint16_t*,uint16_t*,int,int,int,float*,void*);
// Successful host submissions include capture; graph replay does not increment.
unsigned long long attention_rows64_early_qpad8_submissions();
}
