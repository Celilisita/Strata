#include "strata/qwen35moe/runtime.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"

namespace strata::qwen35moe {
// One model layer executor for prefill and speculative verification. Placement
// and token count change the schedule, never the layer's arithmetic or weights.
void Runtime::decode_batch_layer(size_t il, int count, int first_position,
                                  const std::string& reference, Metrics& metrics) {
    auto& b = batch_; auto& layer = layers_[il];
    if (paired_session_) paired_layer_ = il;
    dense_profile_.begin(prefill_event_offset_ + il, !decode_phase_, stream_);
    const bool probe = il == 0 && !reference.empty();
    if (dense_profile_.active() && (probe || paired_session_))
        throw std::invalid_argument("dense profile requires ordinary real prefill");
    kernels::native_qsa_rms_norm_weighted(b.hidden.as<float>(), layer.attn_norm.f32(),
        b.norm.as<float>(), 2048, count, float(spec_.rms_epsilon), stream_);
    dense_profile_.launches(1);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("norm_input", b.norm.data(), size_t(count) * 2048 * 4, false, stream_);
    if (probe) batch_probe_.norm = batch_stage_error(reference, "attn_norm-0", b.norm.as<float>(), 2048);
    if (decode_phase_) batch_quantize_q8(b.norm.as<float>(), count, 2048);
    else batch_quantize_to_half(b.norm.as<float>(), count, 2048);
    dense_profile_.mark(DenseStage::Input, stream_);
    if (spec_.layers[il].recurrent) batch_gdn(layer, count, probe ? reference : std::string());
    else batch_attention(layer, count, first_position);
    kernels::add_inplace(b.mixer.as<float>(), b.hidden.as<float>(), int64_t(count) * 2048, stream_);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("attn_residual", b.mixer.data(), size_t(count) * 2048 * 4, false, stream_);
    if (probe) batch_probe_.residual = batch_stage_error(reference, "attn_residual-0", b.mixer.as<float>(), 2048);
    kernels::native_qsa_rms_norm_weighted(b.mixer.as<float>(), layer.post_norm.f32(),
        b.norm.as<float>(), 2048, count, float(spec_.rms_epsilon), stream_);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("post_norm", b.norm.data(), size_t(count) * 2048 * 4, false, stream_);
    if (probe) batch_probe_.post_norm = batch_stage_error(reference, "attn_post_norm-0", b.norm.as<float>(), 2048);
    dense_profile_.launches(2);
    dense_profile_.mark(DenseStage::Residual, stream_);
    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][1], stream_), "prefill MoE begin");
    batch_moe(il, layer, count, probe ? reference : std::string(), metrics);
    if (probe) {
        batch_probe_.logits = batch_stage_error(reference, "ffn_moe_logits-0", b.logits.as<float>(), 256);
        batch_probe_.layer = batch_stage_error(reference, "l_out-0", b.hidden.as<float>(), 2048);
    }
    if (checkpoint_sink_)
        checkpoint_sink_->gdn_detail("hidden", b.hidden.data(), size_t(count) * 2048 * 4, false, stream_);
    if (checkpoint_sink_)
        for (int t = 0; t < count; ++t)
            checkpoint_sink_->layer(paired_session_ ? paired_positions_[size_t(t)] : first_position + t, il, b.hidden.as<float>() + size_t(t) * 2048, stream_, metrics);
}
} // namespace strata::qwen35moe
