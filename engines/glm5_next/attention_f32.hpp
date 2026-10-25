#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdint>
namespace strata::glm5_next {
constexpr int attention_f32_tag=0x474c4632;
constexpr int attention_f32_tag_index=12;
constexpr int attention_f32_heads_index=13;
constexpr size_t attention_f32_heads32_shared_bytes=78336;
inline bool is_attention_f32(const ggml_tensor* tensor) {
    return tensor->op==GGML_OP_FLASH_ATTN_EXT&&tensor->op_params[attention_f32_tag_index]==attention_f32_tag;
}
inline bool attention_f32_requested() {
    const char* flag=std::getenv("STRATA_GLM53_F32_FLASH");
    if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1"))throw std::runtime_error("GLM F32 fused attention must be0 or1");
    return true;
}
// Host eager/capture dispatches only, never CUDA graph replay or token counts.
struct AttentionF32DispatchCounts {uint64_t f32_dispatches,f16_dispatches,f16_heads32_dispatches;};
GGML_BACKEND_API AttentionF32DispatchCounts attention_f32_dispatch_counts();
// Graph construction only. The retained source dtype snapshots this choice;
// neither graph replay nor a tile load reads process environment.
inline bool attention_f32_kv_direct_f16_requested() {
    const char* flag=std::getenv("STRATA_GLM53_F32_KV_DIRECT_F16");
    if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1"))throw std::runtime_error("GLM direct F16 KV must be0 or1");
    return true;
}
// Only graph construction reads this flag. Slot13 snapshots32, or zero for16.
inline bool attention_f32_kv_heads32_requested() {
    const char* flag=std::getenv("STRATA_GLM53_F32_KV_HEADS32");
    if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1"))throw std::runtime_error("GLM F32 KV heads32 must be0 or1");
    return true;
}
inline bool attention_f32_f16_source_layout(const ggml_tensor* t) {
    if(!t||t->type!=GGML_TYPE_F16||t->ne[0]!=512||t->ne[2]!=1||t->ne[3]!=1||
       t->nb[0]!=sizeof(ggml_fp16_t))return false;
    // Two adjacent stored halves are one half2 load. Keep byte strides rather
    // than flattening a padded context cache or a view into that cache.
    if(t->ne[1]<=0||t->ne[1]>INT32_MAX||t->nb[1]<512*sizeof(ggml_fp16_t)||
       t->nb[1]%4||t->nb[1]>size_t(INT32_MAX)||t->nb[2]>size_t(INT32_MAX)||
       t->nb[3]>size_t(INT64_MAX)||size_t(t->ne[1])>SIZE_MAX/t->nb[1])return false;
    // Axes2/3 are singleton: real [512,1,N] -> permute can leave nb2=1024.
    // Their unused strides need not span all N rows; byte offsets stay zero.
    for(auto* view=t;view;view=view->view_src)if(view->view_offs%4)return false;
    return !t->data||reinterpret_cast<uintptr_t>(t->data)%4==0;
}
inline bool attention_f32_kv_direct_f16_eligible(const ggml_tensor* q,const ggml_tensor* k,const ggml_tensor* v) {
    if(!q||q->type!=GGML_TYPE_F32||q->ne[0]!=512||(q->ne[1]!=1&&q->ne[1]!=2)||
       q->ne[2]!=64||q->ne[3]!=1||q->nb[0]!=sizeof(float))return false;
    if(!attention_f32_f16_source_layout(k)||!attention_f32_f16_source_layout(v))return false;
    if(k!=v&&(!v->view_src||v->view_src!=k->view_src||v->view_offs!=k->view_offs))return false;
    for(int d=0;d<4;++d)if(k->ne[d]!=v->ne[d]||k->nb[d]!=v->nb[d])return false;
    return true;
}
// A Q8 row holds16 blocks (34 bytes each), never512 half scalars.
inline bool attention_f32_q8_source_layout(const ggml_tensor* t) {
    if(!t||t->type!=GGML_TYPE_Q8_0||t->ne[0]!=512||t->ne[2]!=1||t->ne[3]!=1||
       t->nb[0]!=ggml_type_size(GGML_TYPE_Q8_0))return false;
    const size_t block=ggml_type_size(GGML_TYPE_Q8_0);
    if(t->ne[1]<=0||t->ne[1]>INT32_MAX||t->nb[1]<ggml_row_size(GGML_TYPE_Q8_0,512)||
       t->nb[1]%block||t->nb[1]>size_t(INT32_MAX)||t->nb[2]>size_t(INT32_MAX)||
       t->nb[3]>size_t(INT64_MAX)||size_t(t->ne[1])>SIZE_MAX/t->nb[1])return false;
    for(auto* view=t;view;view=view->view_src)if(view->view_offs%block)return false;
    return !t->data||reinterpret_cast<uintptr_t>(t->data)%alignof(uint16_t)==0;
}
// A window is identified by its actual CUDA-compatible storage owner, never
// by a process flag. Logical rows remain complete mapped host history.
inline bool attention_f32_kv_window_source(const ggml_tensor* t) {
    for (; t; t=t->view_src) if (t->buffer) {
        const auto type=ggml_backend_buffer_get_type(t->buffer);
        if (std::strstr(ggml_backend_buft_name(type),"STRATA_GLM_KV_STREAM")) return true;
    }
    return false;
}
inline bool attention_f32_kv_window_eligible(const ggml_tensor* q,const ggml_tensor* k,const ggml_tensor* v) {
    if (!q || q->type!=GGML_TYPE_F32 || q->ne[0]!=512 || q->ne[1]<1 || q->ne[1]>8192 ||
        q->ne[2]!=64 || q->ne[3]!=1 || q->nb[0]!=sizeof(float)) return false;
    const bool f16=attention_f32_f16_source_layout(k)&&attention_f32_f16_source_layout(v);
    const bool q8=attention_f32_q8_source_layout(k)&&attention_f32_q8_source_layout(v);
    if (!f16&&!q8) return false;
    if (!attention_f32_kv_window_source(k) || !attention_f32_kv_window_source(v)) return false;
    if (k!=v && (!v->view_src || v->view_src!=k->view_src || v->view_offs!=k->view_offs)) return false;
    for (int d=0;d<4;++d) if (k->ne[d]!=v->ne[d] || k->nb[d]!=v->nb[d]) return false;
    return true;
}
inline ggml_tensor* build_attention_f32(ggml_context* ctx,ggml_tensor* q,ggml_tensor* k,
    ggml_tensor* v,ggml_tensor* mask,ggml_tensor* projection,float scale,bool v_trans,int64_t selected_limit=0) {
    const auto valid_cache=[](const ggml_tensor* t){return t->type==GGML_TYPE_F32||t->type==GGML_TYPE_F16||
        (t->type==GGML_TYPE_Q8_0&&attention_f32_kv_window_source(t));};
    if(q->type!=GGML_TYPE_F32||!valid_cache(k)||!valid_cache(v)||
       q->ne[0]!=512||k->ne[0]!=512||q->ne[2]!=64||k->ne[2]!=1||q->ne[3]!=1||k->ne[3]!=1)
        throw std::runtime_error(std::string("GLM fused attention geometry: q=")+ggml_type_name(q->type)+
            "/"+std::to_string(q->ne[0])+","+std::to_string(q->ne[1])+","+std::to_string(q->ne[2])+","+std::to_string(q->ne[3])+
            " k="+ggml_type_name(k->type)+"/"+std::to_string(k->ne[0])+","+std::to_string(k->ne[1])+","+
            std::to_string(k->ne[2])+","+std::to_string(k->ne[3])+" v="+ggml_type_name(v->type));
    if(v_trans)v=ggml_transpose(ctx,v);
    bool alias=v==k;
    if(!alias&&v->view_src&&v->view_src==k->view_src&&v->view_offs==k->view_offs) {
        alias=true;for(int dim=0;dim<4;++dim)alias&=v->ne[dim]==k->ne[dim]&&v->nb[dim]==k->nb[dim];
    }
    // Preserve the configured KV storage. Widening stored F16 is exact and
    // the aliased latent needs one conversion, shared by K and V.
    const bool heads32_requested=attention_f32_kv_heads32_requested();
    const bool window_source=attention_f32_kv_window_source(k)||attention_f32_kv_window_source(v);
    const bool window_direct=window_source&&attention_f32_kv_window_eligible(q,k,v);
    if (window_source&&!window_direct) throw std::runtime_error("GLM dynamic KV attention source geometry differs");
    const bool direct_source=window_direct||(attention_f32_kv_direct_f16_requested()&&attention_f32_kv_direct_f16_eligible(q,k,v));
    if(!direct_source) {
    if(k->type==GGML_TYPE_F16)k=ggml_cast(ctx,k,GGML_TYPE_F32);
    if(alias)v=k;else if(v->type==GGML_TYPE_F16)v=ggml_cast(ctx,v,GGML_TYPE_F32);
    if(v->nb[0]!=sizeof(float))v=ggml_cont(ctx,v);
    }
    // Sparse/causal mask entries are zero or negative infinity; F16 is exact.
    auto* result=ggml_flash_attn_ext(ctx,q,k,v,ggml_cast(ctx,mask,GGML_TYPE_F16),scale,0.f,0.f);
    if(selected_limit<0)throw std::runtime_error("GLM attention selection bound is negative");
    const int64_t bound=selected_limit?std::min(selected_limit,k->ne[1]):k->ne[1];
    ggml_flash_attn_ext_set_n_kv_max(result,int32_t(bound));
    ggml_prec_set_acc(result,GGML_PREC_F32);
    result->op_params[attention_f32_tag_index]=attention_f32_tag;
    if(heads32_requested&&direct_source&&!window_source)result->op_params[attention_f32_heads_index]=32;
    if(projection) {
        result=ggml_permute(ctx,result,0,2,1,3);
        result=ggml_mul_mat(ctx,projection,result);
        result=ggml_permute(ctx,result,0,2,1,3);
    }
    return ggml_cont_2d(ctx,result,result->ne[0]*result->ne[1],q->ne[1]);
}
}

struct ggml_backend_cuda_context;
void strata_glm_attention_f32(ggml_backend_cuda_context&,ggml_tensor*);
