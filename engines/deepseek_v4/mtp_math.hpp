#pragma once
#include "mtp_sidecar.hpp"
#include "ggml.h"
#include <array>
#include <cmath>
#include <cstdlib>
namespace strata::deepseek_v4 {
inline ggml_tensor* mtp_trace(ggml_tensor* tensor,const char* name){ggml_set_name(tensor,name);if(std::getenv("STRATA_DSV4_MTP_COMPONENT_TRACE"))ggml_set_output(tensor);return tensor;}
struct MtpHc {ggml_tensor* mixed;ggml_tensor* post;ggml_tensor* comb;};
inline ggml_tensor* mtp_norm(ggml_context* c,ggml_tensor* x,ggml_tensor* weight){return ggml_mul(c,ggml_rms_norm(c,x,1e-6f),weight);}
inline ggml_tensor* mtp_front(ggml_context* c,MtpSidecar& w,ggml_tensor* embedding,ggml_tensor* hidden) {
    auto* e=ggml_mul_mat(c,w.tensor("mtp.0.e_proj.weight"),mtp_norm(c,embedding,w.tensor("mtp.0.enorm.weight")));
    auto* h=ggml_mul_mat(c,w.tensor("mtp.0.h_proj.weight"),mtp_norm(c,hidden,w.tensor("mtp.0.hnorm.weight")));
    return mtp_trace(ggml_add(c,h,ggml_reshape_3d(c,e,4096,1,embedding->ne[1])),"strata_mtp_front");
}
inline ggml_tensor* mtp_hc_reduce(ggml_context* c,ggml_tensor* x,ggml_tensor* weights) {
    return ggml_dsv4_hc_pre(c,x,weights);
}
inline MtpHc mtp_hc_pre(ggml_context* c,MtpSidecar& w,ggml_tensor* x,const char* stage) {
    const std::string root=std::string("mtp.0.hc_")+stage;
    const int64_t tokens=x->ne[2];
    auto* flat=ggml_reshape_2d(c,x,16384,tokens);
    auto* mixes=ggml_mul_mat(c,w.tensor(root+"_fn"),ggml_rms_norm(c,flat,1e-6f));
    auto* scale=w.tensor(root+"_scale");auto* base=w.tensor(root+"_base");
    auto part=[&](int first,int rows,int factor){
        auto* z=ggml_view_2d(c,mixes,rows,tokens,mixes->nb[1],first*sizeof(float));
        auto* a=ggml_view_1d(c,scale,1,factor*sizeof(float));auto* b=ggml_view_1d(c,base,rows,first*sizeof(float));
        return ggml_add(c,ggml_mul(c,z,a),b);};
    auto* pre=ggml_scale_bias(c,ggml_sigmoid(c,part(0,4,0)),1,1e-6f);
    auto* post=ggml_scale(c,ggml_sigmoid(c,part(4,4,1)),2);
    auto* comb=ggml_dsv4_hc_comb(c,mixes,scale,base,1e-6f,20);
    return {mtp_hc_reduce(c,x,pre),post,comb};
}
inline ggml_tensor* mtp_hc_post(ggml_context* c,ggml_tensor* x,ggml_tensor* residual,const MtpHc& hc) {
    return ggml_dsv4_hc_post(c,x,residual,hc.post,hc.comb);
}
inline ggml_tensor* mtp_head_hidden(ggml_context* c,MtpSidecar& w,ggml_tensor* hidden) {
    auto* flat=ggml_reshape_2d(c,hidden,16384,hidden->ne[2]);
    auto* mixes=ggml_mul_mat(c,w.tensor("mtp.0.hc_head_fn"),ggml_rms_norm(c,flat,1e-6f));
    auto* weights=ggml_add(c,ggml_mul(c,mixes,w.tensor("mtp.0.hc_head_scale")),w.tensor("mtp.0.hc_head_base"));
    weights=ggml_scale_bias(c,ggml_sigmoid(c,weights),1,1e-6f);
    return mtp_norm(c,mtp_hc_reduce(c,hidden,weights),w.tensor("mtp.0.norm.weight"));
}
}
