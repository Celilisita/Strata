#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"

namespace strata::qwen35moe {

// Strata's chunk schedule, using the same canonical token expressions and state.
void Runtime::batch_gdn_scan(LayerWeights& layer, int count) {
    auto& b = batch_;
    auto* activated = b.scan_activated.as<float>();
    kernels::native_gdn_conv_scan(layer.history.as<float>(), b.qkv.as<float>(),
        layer.conv.f32(), activated, 8192, count, stream_, verify_history_);
    if (checkpoint_sink_) checkpoint_sink_->gdn_norm_input(activated, count, stream_);
    gdn_l2_norm(activated, 32, 128, float(spec_.rms_epsilon), stream_, count, 8192, prefill_config_.gdn_warp_norm);
    kernels::native_gdn_beta_gate(b.beta.as<float>(), count * 32, stream_);
    kernels::native_gdn_gate_scan(b.alpha.as<float>(), layer.dt.f32(), layer.a.f32(),
        b.scan_gate.as<float>(), 32, count, stream_);
    if (checkpoint_sink_) {
        checkpoint_sink_->gdn_detail("activated", activated, size_t(count) * 8192 * 4, false, stream_);
        checkpoint_sink_->gdn_detail("scan_gate", b.scan_gate.data(), size_t(count) * 32 * 4, false, stream_);
        checkpoint_sink_->gdn_detail("beta_gated", b.beta.data(), size_t(count) * 32 * 4, false, stream_);
    }
    dense_profile_.launches(4);
    dense_profile_.mark(DenseStage::Preprocess, stream_);
    dense_profile_.mark(DenseStage::ShortAttention, stream_);
    kernels::native_gdn_scan(layer.state.as<float>(), activated, activated + 2048,
        activated + 4096, b.scan_gate.as<float>(), b.beta.as<float>(),
        b.scan_core.as<float>(), {128, 16, 32}, count, 8192, stream_, verify_state_, !decode_phase_ && count >= 1024 && prefill_config_.gdn_columns ? 2 : 1);
    if (checkpoint_sink_) {
        checkpoint_sink_->gdn_detail("state", layer.state.data(), layer.state.bytes(), false, stream_);
        checkpoint_sink_->gdn_detail("history", layer.history.data(), layer.history.bytes(), false, stream_);
    }
    dense_profile_.launches(1);
    dense_profile_.mark(DenseStage::Core, stream_);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("gdn_core", b.scan_core.data(), size_t(count) * 4096 * 4, false, stream_);
    gdn_post_silu(b.scan_core.as<float>(), b.z.as<float>(), layer.ssm_norm.f32(),
        b.post.as<float>(), count * 32, float(spec_.rms_epsilon), stream_);
    dense_profile_.launches(1);
    dense_profile_.mark(DenseStage::Post, stream_);
}

} // namespace strata::qwen35moe
