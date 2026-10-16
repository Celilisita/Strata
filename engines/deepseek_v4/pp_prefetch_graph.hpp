#pragma once
#include "pp_cache_view.hpp"
#include "packed_expert_store.hpp"
#include <cstring>
#include <cstdlib>
namespace strata::deepseek_v4 {
inline constexpr uint32_t pp_prefetch_magic=0x50504631;
inline bool pp_prefetch_enabled(){const char* value=std::getenv("STRATA_DSV4_PP_LAYER_PREFETCH");
    return packed_experts_requested()&&!(value&&std::strcmp(value,"0")==0);}
inline void tag_pp_prefetch(ggml_tensor* norm,const PpCacheView* view,int layer,int tokens){
    if(tokens<2048||!view||!pp_prefetch_enabled())return;
    if(norm->op==GGML_OP_MUL&&norm->src[0]&&norm->src[0]->op==GGML_OP_RMS_NORM)norm=norm->src[0];
    if(norm->op!=GGML_OP_RMS_NORM)return;
    // RMS epsilon occupies the first word. Reserved words carry only a native
    // scheduling marker; norm arithmetic and tensor ownership stay unchanged.
    std::memcpy(norm->op_params+4,&view,sizeof(view));norm->op_params[6]=layer;norm->op_params[15]=pp_prefetch_magic;
}
inline const PpCacheView* pp_prefetch_view(const ggml_tensor* node,int& layer){
    if(node->op!=GGML_OP_RMS_NORM||uint32_t(node->op_params[15])!=pp_prefetch_magic)return nullptr;
    const PpCacheView* view=nullptr;std::memcpy(&view,node->op_params+4,sizeof(view));layer=node->op_params[6];return view;
}
}
