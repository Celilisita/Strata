#include "session_state.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::batch_gdn_sessions(LayerWeights& layer, int count) {
    if (!paired_session_ || count != 2) throw std::invalid_argument("independent GDN rows require two sessions");
    auto& b = batch_; auto& other = paired_session_->layers[paired_layer_];
    auto* activated = b.scan_activated.as<float>();
    for (int t = 0; t < 2; ++t) {
        float* history = t ? other.history.as<float>() : layer.history.as<float>();
        kernels::native_gdn_conv_scan(history, b.qkv.as<float>() + size_t(t) * 8192,
            layer.conv.f32(), activated + size_t(t) * 8192, 8192, 1, stream_);
    }
    gdn_l2_norm(activated, 32, 128, float(spec_.rms_epsilon), stream_, 2, 8192, prefill_config_.gdn_warp_norm);
    kernels::native_gdn_beta_gate(b.beta.as<float>(), 64, stream_);
    kernels::native_gdn_gate_scan(b.alpha.as<float>(), layer.dt.f32(), layer.a.f32(),
        b.scan_gate.as<float>(), 32, 2, stream_);
    for (int t = 0; t < 2; ++t) {
        float* state = t ? other.state.as<float>() : layer.state.as<float>();
        const auto* row = activated + size_t(t) * 8192;
        kernels::native_gdn_scan(state, row, row + 2048, row + 4096,
            b.scan_gate.as<float>() + size_t(t) * 32, b.beta.as<float>() + size_t(t) * 32,
            b.scan_core.as<float>() + size_t(t) * 4096, {128, 16, 32}, 1, 8192, stream_);
    }
    gdn_post_silu(b.scan_core.as<float>(), b.z.as<float>(), layer.ssm_norm.f32(),
        b.post.as<float>(), 64, float(spec_.rms_epsilon), stream_);
}
} // namespace strata::qwen35moe
