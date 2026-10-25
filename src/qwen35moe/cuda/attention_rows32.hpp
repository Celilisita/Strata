#pragma once
#include "strata/qwen35moe/attention_tc.hpp"
namespace strata::qwen35moe {
struct AttentionRows32Resources {
    int rows,threads,registers,active_blocks,binary_version,reason_mask;
    size_t static_shared,dynamic_shared,local_bytes,optin_limit;
};
// The caller owns original geometry, Reuse and Register selection.
bool attention_rows32_selected(int first,int count);
AttentionRows32Resources attention_rows32_resources();
// False is a resource fallback before any KV/output mutation; CUDA errors throw.
bool attention_rows32_prefill(const float*,uint16_t*,uint16_t*,int,int,int,float*,void*);
}
