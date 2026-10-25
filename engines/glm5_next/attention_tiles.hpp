#pragma once
#include "ggml.h"
#include "attention_f32.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::glm5_next {
// Q/K/V are already permuted by build_attn_mha. Retain F32 latent math and
// the full selected-key mask, bounding only simultaneous query-score storage.
inline ggml_tensor* build_attention_tiles(ggml_context* ctx, ggml_cgraph* graph,
    ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask,
    ggml_tensor* v_mla, float scale, bool v_trans,int64_t selected_limit=0) {
    if(q->ne[3]!=1 || k->ne[3]!=1 || !mask || mask->ne[1]<q->ne[1] ||
       mask->ne[2]!=1 || mask->ne[3]!=1 || v_mla->ne[0]!=512)
        throw std::runtime_error("GLM F32 attention tile geometry differs");
    if(attention_f32_requested() || attention_f32_kv_window_source(k)) {
        auto* result=build_attention_f32(ctx,q,k,v,mask,v_mla,scale,v_trans,selected_limit);
        ggml_build_forward_expand(graph,result);return result;
    }
    if(!v_trans) v=ggml_cont(ctx,ggml_transpose(ctx,v));
    ggml_tensor* result=nullptr;
    for(int64_t first=0;first<q->ne[1];first+=512) {
        const int64_t count=std::min<int64_t>(512,q->ne[1]-first);
        auto* queries=ggml_view_4d(ctx,q,q->ne[0],count,q->ne[2],1,
            q->nb[1],q->nb[2],q->nb[3],first*q->nb[1]);
        auto* selected=ggml_view_4d(ctx,mask,mask->ne[0],count,1,1,
            mask->nb[1],mask->nb[2],mask->nb[3],first*mask->nb[1]);
        auto* scores=ggml_mul_mat(ctx,k,queries);
        ggml_prec_set_acc(scores,GGML_PREC_F32);
        scores=ggml_soft_max_ext(ctx,scores,selected,scale,0.0f);
        auto* values=ggml_mul_mat(ctx,v,scores);
        values=ggml_mul_mat(ctx,v_mla,values);
        values=ggml_permute(ctx,values,0,2,1,3);
        values=ggml_cont_2d(ctx,values,values->ne[0]*values->ne[1],count);
        // Complete each tile before the next score matrix is allocated.
        ggml_build_forward_expand(graph,values);
        result=result?ggml_concat(ctx,result,values,1):values;
        ggml_build_forward_expand(graph,result);
    }
    return result;
}
}
