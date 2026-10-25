#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_gdn_conv_time32.hpp"
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace strata::qwen35moe {

// Strata's chunk schedule, using the same canonical token expressions and state.
void Runtime::batch_gdn_scan(LayerWeights& layer, int count) {
    auto& b = batch_;
    auto* activated = b.scan_activated.as<float>();
    static const bool alpha_hoist=[] {const char* raw=std::getenv("STRATA_QWEN35_GDN_ALPHA_HOIST");
        if(!raw||!std::strcmp(raw,"0"))return false;if(!std::strcmp(raw,"1"))return true;
        throw std::invalid_argument("GDN_ALPHA_HOIST requires0 or1");}();
    const bool alpha=alpha_hoist&&count>=1024&&prefill_config_.gdn_scan_columns(count,true)==8&&
        !decode_phase_&&!verify_state_&&!verify_history_&&!checkpoint_sink_&&!paired_session_&&
        !sessions_&&!mtp_enabled_&&!kernels::native_gdn_pair2_enabled()&&!kernels::native_gdn_chunk4_enabled();
    const bool parallel_conv = prefill_config_.gdn_parallel_convolution(count,
        !decode_phase_ && !verify_history_ && !verify_state_ && !checkpoint_sink_ &&
        !paired_session_ && !sessions_ && !mtp_enabled_);
    const bool time32=kernels::native_gdn_conv_time32_selected(!decode_phase_,
        session_capacity()==1&&!paired_session_,mtp_enabled_,8192,count,verify_history_!=nullptr)&&
        kernels::native_gdn_conv_time32(layer.history.as<float>(),b.qkv.as<float>(),
            layer.conv.f32(),activated,8192,count,stream_);
    if (!time32 && parallel_conv)
        kernels::native_gdn_conv_scan_parallel(layer.history.as<float>(), b.qkv.as<float>(),
            layer.conv.f32(), activated, 8192, count, stream_);
    else if (!time32)
        kernels::native_gdn_conv_scan(layer.history.as<float>(), b.qkv.as<float>(),
            layer.conv.f32(), activated, 8192, count, stream_, verify_history_);
    if (checkpoint_sink_) checkpoint_sink_->gdn_norm_input(activated, count, stream_);
    gdn_l2_norm(activated, 32, 128, float(spec_.rms_epsilon), stream_, count, 8192, prefill_config_.gdn_warp_norm);
    kernels::native_gdn_beta_gate(b.beta.as<float>(), count * 32, stream_);
    (alpha?kernels::native_gdn_gate_scan_alpha:kernels::native_gdn_gate_scan)(b.alpha.as<float>(), layer.dt.f32(), layer.a.f32(),
        b.scan_gate.as<float>(), 32, count, stream_);
    if (checkpoint_sink_) {
        checkpoint_sink_->gdn_detail("activated", activated, size_t(count) * 8192 * 4, false, stream_);
        checkpoint_sink_->gdn_detail("scan_gate", b.scan_gate.data(), size_t(count) * 32 * 4, false, stream_);
        checkpoint_sink_->gdn_detail("beta_gated", b.beta.data(), size_t(count) * 32 * 4, false, stream_);
    }
    dense_profile_.launches((parallel_conv || time32) ? 5 : 4);
    dense_profile_.mark(DenseStage::Preprocess, stream_);
    dense_profile_.mark(DenseStage::ShortAttention, stream_);
    // Decode, short scans and prefix snapshots retain the one-column provider.
    const int columns = prefill_config_.gdn_scan_columns(count, !decode_phase_ && !verify_state_);
    const bool block_scan = count >= 1024 && columns == 8 && !decode_phase_ &&
        !verify_state_ && !verify_history_ && !checkpoint_sink_ && !paired_session_ &&
        !sessions_ && !mtp_enabled_;
    const bool chunk4 = block_scan && kernels::native_gdn_chunk4_enabled();
    const bool pair2 = !chunk4 && block_scan && kernels::native_gdn_pair2_enabled();
    if(alpha)
        kernels::native_gdn_scan_alpha(layer.state.as<float>(),activated,activated+2048,
            activated+4096,b.scan_gate.as<float>(),b.beta.as<float>(),b.scan_core.as<float>(),
            {128,16,32},count,8192,stream_);
    else if(chunk4)
        kernels::native_gdn_scan_chunk4(layer.state.as<float>(),activated,activated+2048,
            activated+4096,b.scan_gate.as<float>(),b.beta.as<float>(),b.scan_core.as<float>(),
            {128,16,32},count,8192,b.qkv.data(),b.qkv.bytes(),stream_);
    else if(pair2)
        kernels::native_gdn_scan_pair2(layer.state.as<float>(),activated,activated+2048,
            activated+4096,b.scan_gate.as<float>(),b.beta.as<float>(),b.scan_core.as<float>(),
            {128,16,32},count,8192,b.qkv.data(),b.qkv.bytes(),stream_);
    else {
        kernels::native_gdn_scan(layer.state.as<float>(), activated, activated + 2048,
            activated + 4096, b.scan_gate.as<float>(), b.beta.as<float>(),
            b.scan_core.as<float>(), {128, 16, 32}, count, 8192, stream_, verify_state_, columns);
    }
    if (checkpoint_sink_) {
        checkpoint_sink_->gdn_detail("state", layer.state.data(), layer.state.bytes(), false, stream_);
        checkpoint_sink_->gdn_detail("history", layer.history.data(), layer.history.bytes(), false, stream_);
    }
    dense_profile_.launches(chunk4 || pair2 ? 2 : 1);
    dense_profile_.mark(DenseStage::Core, stream_);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("gdn_core", b.scan_core.data(), size_t(count) * 4096 * 4, false, stream_);
    gdn_post_silu(b.scan_core.as<float>(), b.z.as<float>(), layer.ssm_norm.f32(),
        b.post.as<float>(), count * 32, float(spec_.rms_epsilon), stream_);
    dense_profile_.launches(1);
    dense_profile_.mark(DenseStage::Post, stream_);
}

} // namespace strata::qwen35moe
