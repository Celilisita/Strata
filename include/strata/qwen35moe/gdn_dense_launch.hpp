#pragma once
#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/elementwise.hpp"

namespace strata::qwen35moe {
// Exactly the existing one-token kernels and their arithmetic/order. This helper
// owns nothing and performs no allocation, synchronization, host copy or routing.
template<class Project>
void launch_gdn_mixer(LayerWeights& layer,Workspace& w,float epsilon,bool warp_norm,
                      cudaStream_t stream,Project project){
    const auto q=[](DeviceBuffer& b){return b.as<float>();};
    kernels::native_quantize_q8_1(q(w.norm),w.q8_input.data(),2048,1,stream);
    project(layer.qkv,w.q8_input.data(),q(w.qkv),2048,8192);
    project(layer.gate,w.q8_input.data(),q(w.z),2048,4096);
    project(layer.alpha,w.q8_input.data(),q(w.alpha),2048,32);
    project(layer.beta,w.q8_input.data(),q(w.beta),2048,32);
    kernels::native_gdn_conv_silu(q(layer.history),q(w.qkv),layer.conv.f32(),q(w.raw),q(w.activated),8192,4,stream);
    gdn_l2_norm(q(w.activated),32,128,epsilon,stream,1,0,warp_norm);
    kernels::native_gdn_beta_gate(q(w.beta),32,stream);
    kernels::native_gdn_gate(q(w.alpha),layer.dt.f32(),layer.a.f32(),q(w.gate),32,stream);
    kernels::native_gdn_step(q(layer.state),q(w.activated),q(w.activated)+2048,q(w.activated)+4096,
                           q(w.gate),q(w.beta),q(w.core),{128,16,32},stream);
    gdn_post_silu(q(w.core),q(w.z),layer.ssm_norm.f32(),q(w.post),32,epsilon,stream);
    kernels::native_quantize_q8_1(q(w.post),w.q8_wide.data(),4096,1,stream);
    project(layer.ssm_out,w.q8_wide.data(),q(w.mixer),4096,2048);
}
template<class Project>
void launch_gdn_dense(LayerWeights& layer,Workspace& w,float epsilon,bool warp_norm,
                      cudaStream_t stream,Project project){
    kernels::native_qsa_rms_norm_weighted(w.hidden.as<float>(),layer.attn_norm.f32(),w.norm.as<float>(),2048,1,epsilon,stream);
    launch_gdn_mixer(layer,w,epsilon,warp_norm,stream,project);
    kernels::add_inplace(w.mixer.as<float>(),w.hidden.as<float>(),2048,stream);
    kernels::native_qsa_rms_norm_weighted(w.mixer.as<float>(),layer.post_norm.f32(),w.norm.as<float>(),2048,1,epsilon,stream);
}
}
