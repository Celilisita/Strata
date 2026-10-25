#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::gemma4::native {
inline bool tg_global_gqa_requested(){
    static const bool enabled=[](){
        const char* value=std::getenv("STRATA_GEMMA_TG_GLOBAL_GQA");
        if(!value||std::strcmp(value,"0")==0)return false;
        if(std::strcmp(value,"1")==0)return true;
        throw std::invalid_argument("STRATA_GEMMA_TG_GLOBAL_GQA requires0or1");
    }();return enabled;
}
struct GlobalGqaResources {int registers,local_bytes,static_shared,dynamic_shared,active_CTA_per_SM,SM_shared_bytes,SM_max_threads;};
// Cold kernel attribute/resource preparation only. No projection or forward.
GlobalGqaResources prepare_global_t1_gqa(cudaStream_t);
void split_attention_global_t1_gqa(const float*,float*,CacheView,const int*,int,int,AttentionWorkspace,cudaStream_t);
// Focused checker seam: same AoS partial inventory, no merge or extra owner.
void global_t1_gqa_segment_only(const float*,CacheView,const int*,AttentionWorkspace,cudaStream_t);
}
