#include "strata/qwen35moe/gdn_dense_launch.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_coalesced.hpp"
#include "strata/kernels/native_gdn_cluster.hpp"
#include "tg_gdn_cluster.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"

namespace strata::qwen35moe {
namespace {
bool coalesced_gdn_enabled() {
    const char* raw=std::getenv("STRATA_QWEN35_TG_GDN_COALESCED");
    if(!raw||std::strcmp(raw,"0")==0)return false;
    if(std::strcmp(raw,"1")==0)return true;
    throw std::invalid_argument("TG_GDN_COALESCED must be0 or1");
}
}

void Runtime::gdn_step(LayerWeights& layer) {
    const bool coalesced=coalesced_gdn_enabled();
    auto& w = work_;
    const auto q = [&](DeviceBuffer& buffer) { return buffer.as<float>(); };
    kernels::native_quantize_q8_1(q(w.norm), w.q8_input.data(), 2048, 1, stream_);
    project(layer.qkv, w.q8_input.data(), q(w.qkv), 2048, 8192);
    project(layer.gate, w.q8_input.data(), q(w.z), 2048, 4096);
    project(layer.alpha, w.q8_input.data(), q(w.alpha), 2048, 32);
    project(layer.beta, w.q8_input.data(), q(w.beta), 2048, 32);
    const bool cluster=detail::tg_gdn_cluster_eligible(decode_phase_,tg_dynamic_,session_capacity(),
        paired_session_,mtp_enabled_,checkpoint_sink_||mtp_sink_,prefill_config_.gdn_warp_norm)&&
        detail::tg_gdn_cluster_enabled();
    if(!(cluster&&kernels::native_gdn_cluster_step(q(layer.history),q(w.qkv),layer.conv.f32(),
        q(w.raw),q(w.activated),q(w.beta),q(w.alpha),layer.dt.f32(),layer.a.f32(),q(w.gate),
        float(spec_.rms_epsilon),stream_))) {
    kernels::native_gdn_conv_silu(q(layer.history), q(w.qkv), layer.conv.f32(),
                                   q(w.raw), q(w.activated), 8192, 4, stream_);
    gdn_l2_norm(q(w.activated), 32, 128, float(spec_.rms_epsilon), stream_, 1, 0, prefill_config_.gdn_warp_norm);
    kernels::native_gdn_beta_gate(q(w.beta), 32, stream_);
    kernels::native_gdn_gate(q(w.alpha), layer.dt.f32(), layer.a.f32(),
                             q(w.gate), 32, stream_);
    }
    const bool use_coalesced=coalesced&&decode_phase_&&session_capacity()==1&&!paired_session_&&!mtp_enabled_&&
        spec_.n_embd==2048&&spec_.ssm_state==128&&spec_.ssm_group==16&&spec_.ssm_rank==32&&spec_.ssm_inner==4096;
    if(!(use_coalesced&&kernels::native_gdn_coalesced_step(q(layer.state),q(w.activated),q(w.activated)+2048,
          q(w.activated)+4096,q(w.gate),q(w.beta),q(w.core),{128,16,32},stream_))) {
    kernels::native_gdn_step(q(layer.state), q(w.activated), q(w.activated) + 2048,
                             q(w.activated) + 4096, q(w.gate), q(w.beta), q(w.core),
                             {128, 16, 32}, stream_);
    }
    gdn_post_silu(q(w.core), q(w.z), layer.ssm_norm.f32(), q(w.post),
                  32, float(spec_.rms_epsilon), stream_);
    kernels::native_quantize_q8_1(q(w.post), w.q8_wide.data(), 4096, 1, stream_);
    project(layer.ssm_out, w.q8_wide.data(), q(w.mixer), 4096, 2048);
}

} // namespace strata::qwen35moe
