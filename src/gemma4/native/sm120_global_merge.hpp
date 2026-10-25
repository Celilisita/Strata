#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cstdlib>
#include <cstring>
namespace strata::gemma4::native {
inline bool tg_global_merge_requested(){
    const char* value=std::getenv("STRATA_GEMMA_TG_GLOBAL_MERGE");
    if(!value||std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::invalid_argument("Gemma TG global merge must be explicit0or1");
}
inline bool tg_local270_requested(){
    const char* value=std::getenv("STRATA_GEMMA_TG_LOCAL270");
    if(!value||std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::invalid_argument("Gemma TG local270 must be explicit0or1");
}
// Only global Q32/KV4/D512/T1. Both variants use the same original segment.
void split_attention_global_t1_merge(const float*,float*,CacheView,const int*,int,int,AttentionWorkspace,cudaStream_t);
// Private focused-check seams: existing AoS arena, no segment/transfer/allocation.
void global_t1_merge_only(float*,const int*,int capacity,AttentionWorkspace,cudaStream_t);
void global_t1_merge_reference_only(float*,const int*,int capacity,AttentionWorkspace,cudaStream_t);
}
