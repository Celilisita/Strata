#pragma once
#include "mtp_math.hpp"
namespace strata::deepseek_v4 {
struct MtpPrepared {
    ggml_tensor *residual,*activation,*ids,*weights,*shared,*next_kv;
    MtpHc hc;
};
inline ggml_tensor* mtp_input_kv(ggml_context* c,MtpSidecar& w,ggml_tensor* embedding,
    ggml_tensor* ids,ggml_tensor* hidden,ggml_tensor* positions,ggml_tensor* embedded=nullptr) {
    auto* residual=mtp_front(c,w,embedded?embedded:ggml_get_rows(c,embedding,ids),hidden);
    auto hc=mtp_hc_pre(c,w,residual,"attn");
    auto* x=mtp_norm(c,hc.mixed,w.tensor("mtp.0.attn_norm.weight"));
    auto* kv=mtp_norm(c,ggml_mul_mat(c,w.tensor("mtp.0.attn.wkv.weight"),x),w.tensor("mtp.0.attn.kv_norm.weight"));
    kv=ggml_rope_ext(c,ggml_reshape_3d(c,kv,512,1,ids->ne[0]),positions,nullptr,64,0,0,10000.f,1.f,0.f,1.f,0.f,0.f);
    return ggml_reshape_2d(c,ggml_rope_set_offset(kv,448),512,ids->ne[0]);
}
inline ggml_tensor* mtp_attention(ggml_context* c,MtpSidecar& w,ggml_tensor* x,
    ggml_tensor* positions,ggml_tensor* past_kv,ggml_tensor* mask,ggml_tensor** next_kv) {
    const int64_t tokens=x->ne[1];
    auto* qr=mtp_norm(c,ggml_mul_mat(c,w.tensor("mtp.0.attn.wq_a.weight"),x),w.tensor("mtp.0.attn.q_norm.weight"));
    auto* q=ggml_mul_mat(c,w.tensor("mtp.0.attn.wq_b.weight"),qr);
    q=ggml_rms_norm(c,ggml_reshape_3d(c,q,512,64,tokens),1e-6f);
    auto rope=[&](ggml_tensor* z,bool inverse){auto* result=inverse?
        ggml_rope_ext_back(c,z,positions,nullptr,64,0,0,10000.f,1.f,0.f,1.f,0.f,0.f):
        ggml_rope_ext(c,z,positions,nullptr,64,0,0,10000.f,1.f,0.f,1.f,0.f,0.f);
        return ggml_rope_set_offset(result,448);};
    q=mtp_trace(rope(q,false),"strata_mtp_q");
    auto* kv=mtp_norm(c,ggml_mul_mat(c,w.tensor("mtp.0.attn.wkv.weight"),x),w.tensor("mtp.0.attn.kv_norm.weight"));
    kv=mtp_trace(rope(ggml_reshape_3d(c,kv,512,1,tokens),false),"strata_mtp_kv");
    kv=ggml_reshape_2d(c,kv,512,tokens);
    *next_kv=past_kv?ggml_concat(c,past_kv,kv,1):kv;
    const int64_t cells=(*next_kv)->ne[1];
    // MTP has a 128-cell, uncompressed window. Heads are columns in one
    // dense GEMM; the learned sink contributes only to the softmax denominator.
    auto* scores=ggml_mul_mat(c,*next_kv,ggml_reshape_2d(c,q,512,64*tokens));
    scores=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,scores,cells,64,tokens),0,2,1,3));
    auto* causal=mask?ggml_cont(c,ggml_view_2d(c,mask,cells,tokens,mask->nb[1],0)):nullptr;
    auto* probabilities=ggml_soft_max_ext(c,scores,causal,1.f/std::sqrt(512.f),0);
    ggml_soft_max_add_sinks(probabilities,w.tensor("mtp.0.attn.attn_sink"));
    auto* values=ggml_cont(c,ggml_transpose(c,*next_kv));
    auto* out=ggml_cont(c,ggml_permute(c,ggml_mul_mat(c,values,probabilities),0,2,1,3));
    mtp_trace(out,"strata_mtp_attention_value");
    out=rope(ggml_reshape_3d(c,out,512,64,tokens),true);
    out=ggml_permute(c,ggml_reshape_3d(c,out,4096,8,tokens),0,2,1,3);
    auto* wo_a=ggml_reshape_3d(c,w.tensor("mtp.0.attn.wo_a.weight"),4096,1024,8);
    auto* oa=ggml_mul_mat(c,wo_a,out);oa=ggml_cont_2d(c,ggml_permute(c,oa,0,2,1,3),8192,tokens);
    return mtp_trace(ggml_mul_mat(c,w.tensor("mtp.0.attn.wo_b.weight"),oa),"strata_mtp_attention");
}
inline MtpPrepared mtp_prepare(ggml_context* c,MtpSidecar& w,ggml_tensor* shared_embedding,
    ggml_tensor* token_ids,ggml_tensor* hidden,ggml_tensor* positions,ggml_tensor* past_kv,ggml_tensor* mask,ggml_tensor* embedded=nullptr) {
    auto* embedding=embedded?embedded:ggml_get_rows(c,shared_embedding,token_ids);
    auto* residual=mtp_front(c,w,embedding,hidden);
    const auto attention_hc=mtp_hc_pre(c,w,residual,"attn");
    auto* attention=mtp_attention(c,w,mtp_norm(c,attention_hc.mixed,w.tensor("mtp.0.attn_norm.weight")),positions,past_kv,mask,&past_kv);
    residual=mtp_trace(mtp_hc_post(c,attention,residual,attention_hc),"strata_mtp_after_attention");
    const auto hc=mtp_hc_pre(c,w,residual,"ffn");
    mtp_trace(hc.mixed,"strata_mtp_ffn_mixed");
    auto* x=mtp_trace(mtp_norm(c,hc.mixed,w.tensor("mtp.0.ffn_norm.weight")),"strata_mtp_ffn_norm");
    auto* scores=ggml_sqrt(c,ggml_softplus(c,ggml_mul_mat(c,w.tensor("mtp.0.ffn.gate.weight"),x)));
    auto* ids=ggml_top_k(c,ggml_add(c,scores,w.tensor("mtp.0.ffn.gate.bias")),6);
    auto* weights=ggml_get_rows(c,ggml_reshape_3d(c,scores,1,256,x->ne[1]),ids);
    weights=ggml_reshape_2d(c,weights,6,x->ne[1]);weights=ggml_scale(c,ggml_div(c,weights,ggml_sum_rows(c,weights)),1.5f);
    weights=ggml_reshape_3d(c,weights,1,6,x->ne[1]);
    auto* gate=ggml_clamp(c,ggml_mul_mat(c,w.tensor("mtp.0.ffn.shared_experts.w1.weight"),x),-INFINITY,10.f);
    auto* up=ggml_clamp(c,ggml_mul_mat(c,w.tensor("mtp.0.ffn.shared_experts.w3.weight"),x),-10.f,10.f);
    auto* shared=ggml_mul_mat(c,w.tensor("mtp.0.ffn.shared_experts.w2.weight"),ggml_mul(c,ggml_silu(c,gate),up));
    return {residual,x,ids,weights,shared,past_kv,hc};
}
inline ggml_tensor* mtp_finish(ggml_context* c,MtpSidecar& w,const MtpPrepared& p,
    ggml_tensor* contribution,ggml_tensor* shared_head,ggml_tensor** next_hidden=nullptr) {
    ggml_tensor* sum=nullptr;
    auto* weighted=ggml_mul(c,contribution,p.weights);
    for(int rank=0;rank<6;++rank){auto* part=ggml_view_2d(c,weighted,4096,weighted->ne[2],weighted->nb[2],rank*weighted->nb[1]);sum=sum?ggml_add(c,sum,part):part;}
    auto* hidden=mtp_hc_post(c,ggml_add(c,sum,p.shared),p.residual,p.hc);
    if(next_hidden)*next_hidden=hidden;
    return ggml_mul_mat(c,shared_head,mtp_head_hidden(c,w,hidden));
}
}
