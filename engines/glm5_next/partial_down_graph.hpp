#pragma once
#include "partial_down_protocol.hpp"
#include "ggml-backend.h"
#include "../deepseek_v4/device_cache_protocol.hpp"
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {
inline void partial_down_attach_lookup(ggml_tensor* lookup,PartialDownGraphBinding binding){
    if(!binding.bridge&&!binding.indices)return;
    if(!deepseek_v4::device_cache_lookup(lookup)||lookup->op_params[2]!=24||lookup->op_params[10]!=1||lookup->op_params[11]!=1||
        lookup->op_params[cold_gpu_magic_word]!=cold_gpu_op_magic||ggml_nelements(lookup)!=16||lookup->src[8]||lookup->src[9]||
        !partial_down_bridge_tensor(binding.bridge)||!binding.indices||binding.indices->type!=GGML_TYPE_I32||
        ggml_nelements(binding.indices)!=8||!ggml_is_contiguous(binding.indices)||!binding.bridge->data||!binding.indices->data||
        !binding.bridge->buffer||!binding.indices->buffer||ggml_backend_buffer_is_host(binding.bridge->buffer)||
        ggml_backend_buffer_is_host(binding.indices->buffer)||
        (binding.bridge->flags&(GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT))!=(GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT)||
        (binding.indices->flags&(GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT))!=(GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT))
        throw std::runtime_error("partial Down external CUDA binding differs");
    lookup->src[8]=binding.bridge;lookup->src[9]=binding.indices;
}
// build_down uses the existing precision/scale/LoRA-aware MMID builder. The
// retired raw Down is a real dependency before either transient Down row changes.
template<class BuildDown>
inline ggml_tensor* partial_down_build_branch(ggml_context* ctx,ggml_tensor* lookup,ggml_tensor* retired_down,BuildDown build_down){
    if(!lookup||!lookup->src[8])return nullptr;
    if(!partial_down_bridge_tensor(lookup->src[8])||!lookup->src[9]||lookup->src[9]->type!=GGML_TYPE_I32||
        ggml_nelements(lookup->src[9])!=8||!retired_down||retired_down->type!=GGML_TYPE_F32||
        ggml_nelements(retired_down)!=65536||!ggml_is_contiguous(retired_down)||!lookup->src[2]||!lookup->src[7])
        throw std::runtime_error("partial Down branch shape differs");
    auto* mid=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2048,1,4);mid->op=GGML_OP_GET_ROWS;
    mid->src[0]=retired_down;mid->src[1]=lookup;mid->src[2]=lookup->src[8];mid->src[3]=lookup->src[2];mid->src[4]=lookup->src[7];
    std::memcpy(mid->op_params,lookup->op_params,GGML_MAX_OP_PARAMS);mid->op_params[partial_down_tag_word]=partial_down_mid_tag;
    ggml_set_name(mid,"partial_down_mid_after_whole_retired");
    ggml_set_output(mid);
    auto* slots=ggml_view_2d(ctx,lookup->src[9],1,4,sizeof(int32_t),0);
    auto* ids=ggml_view_2d(ctx,lookup->src[9],1,4,sizeof(int32_t),4*sizeof(int32_t));
    auto* out=build_down(mid,ids,slots);
    if(!out||out->type!=GGML_TYPE_F32||out->ne[0]!=4096||out->ne[1]!=1||out->ne[2]!=4||out->ne[3]!=1)
        throw std::runtime_error("partial Down canonical output differs");
    ggml_set_name(out,"partial_down_canonical_four_occurrences");ggml_set_output(out);return out;
}
inline void partial_down_attach_join(ggml_tensor* weighted,ggml_tensor* lookup,ggml_tensor* partial){
    if(!lookup||!lookup->src[8])return;
    if(!deepseek_v4::device_cache_hybrid_weight(weighted)||!partial||ggml_nelements(partial)!=16384||weighted->src[6]||weighted->src[7])
        throw std::runtime_error("partial Down final join dependency differs");
    weighted->src[6]=partial;weighted->src[7]=lookup->src[8];
}
}
