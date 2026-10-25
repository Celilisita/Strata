#pragma once
#include "../deepseek_v4/device_cache_protocol.hpp"
#include "partial_down_graph.hpp"
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
inline ggml_tensor* hybrid_lookup(ggml_tensor* slots) {
    if(slots&&slots->op==GGML_OP_RESHAPE)slots=slots->src[0];
    return deepseek_v4::device_cache_lookup(slots)&&slots->op_params[10]==1?slots:nullptr;
}
inline void hybrid_activation(ggml_tensor* slots,ggml_tensor* activation){
    auto* lookup=hybrid_lookup(slots);if(!lookup)return;
    if(!activation||activation->type!=GGML_TYPE_F32||(ggml_nelements(activation)!=4096&&ggml_nelements(activation)!=8192&&
        !(lookup->op_params[11]==1&&ggml_nelements(lookup)==32&&ggml_nelements(activation)==16384))||!ggml_is_contiguous(activation))
        throw std::runtime_error("GLM hybrid activation differs");
    lookup->src[5]=activation;
}
inline ggml_tensor* hybrid_weight(ggml_context* ctx,ggml_tensor* down,ggml_tensor* weights,ggml_tensor* slots,ggml_tensor* partial=nullptr){
    auto* out=ggml_mul(ctx,down,weights);auto* lookup=hybrid_lookup(slots);if(!lookup)return out;
    if(down->ne[0]!=4096||down->ne[1]!=8||(down->ne[2]!=1&&down->ne[2]!=2&&
        !(lookup->op_params[11]==1&&ggml_nelements(lookup)==32&&down->ne[2]==4))||!ggml_is_contiguous(down)||!ggml_is_contiguous(weights))
        throw std::runtime_error("GLM hybrid contribution shape differs");
    out->src[2]=slots;
    if(lookup->op_params[11]) {
        out->src[3]=lookup->src[7];out->src[4]=lookup->src[2];out->src[5]=lookup->src[3];
    }
    std::memcpy(out->op_params,lookup->op_params,GGML_MAX_OP_PARAMS);
    out->op_params[deepseek_v4::device_cache_tag_index]=deepseek_v4::device_cache_hybrid_weight_tag;
    partial_down_attach_join(out,lookup,partial);
    return out;
}
}
