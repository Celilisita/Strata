#include "strata/qwen35moe/runtime.hpp"
#include "mtp_state.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::set_mtp_execution(bool enabled) {
    if (decode_phase_ || host_pick_.id != -1 || (enabled && !mtp_))
        throw std::invalid_argument("MTP execution must be configured before prefill with resident resources");
    mtp_enabled_ = enabled;
}
void Runtime::initialize_mtp() {
    if (!mtp_enabled_) return;
    const auto& source = spec_.mtp[0];
    auto m = std::make_unique<MtpState>();
    m->decoder = load_layer(source.decoder);
    m->eh_proj.load(gguf_, source.eh_proj);
    m->enorm.load(gguf_, source.enorm); m->hnorm.load(gguf_, source.hnorm);
    m->output_norm.load(gguf_, source.output_norm);
    if (source.output != spec_.output) m->output.load(gguf_, source.output);
    m->concat.reset(size_t((std::min)(1024, max_context_)) * 4096 * sizeof(float));
    m->main_norm.reset(size_t((std::min)(1024, max_context_)) * 2048 * sizeof(float));
    m->seed.reset(size_t(spec_.n_embd) * sizeof(float));
    m->verify_logits.reset(size_t(2) * spec_.n_vocab * sizeof(float));
    m->picks.reset(2 * sizeof(HeadPick));
    m->prefix_state.resize(layers_.size()); m->prefix_history.resize(layers_.size());
    for (size_t il = 0; il < layers_.size(); ++il) {
        if (!spec_.layers[il].recurrent) continue;
        m->prefix_state[il].reset(layers_[il].state.bytes());
        m->prefix_history[il].reset(layers_[il].history.bytes());
    }
    m->embedding_row_bytes = kernels::native_mmvq_weight_bytes(int(source.embedding->type), 2048, 1);
    m->embedding_row.reset(m->embedding_row_bytes);
    mtp_ = std::move(m);
}

// Both draft and catch-up use this conditioning formula. Input main rows must
// be normalized and disjoint from batch hidden/norm/concat scratch.
void Runtime::mtp_condition(const int* tokens, int count, const float* normalized_main,
    int first_position, const std::array<int, 2>* positions) {
    if (!mtp_ || !tokens || !normalized_main || count < 1 || count > batch_.capacity ||
        size_t(count) * 4096 * sizeof(float) > mtp_->concat.bytes())
        throw std::invalid_argument("unprepared MTP conditioning");
    auto& m = *mtp_; auto& b = batch_;
    const auto& source = spec_.mtp[0];
    for (int t = 0; t < count; ++t) {
        if (tokens[t] < 0 || tokens[t] >= int(spec_.n_vocab)) throw std::out_of_range("MTP token");
        load_embedding(tokens[t], positions ? (*positions)[size_t(t)] : first_position + t,
            b.hidden.as<float>() + size_t(t) * 2048, source.embedding,
            m.embedding_row, m.embedding_row_bytes, positions ? t : -1);
    }
    kernels::native_qsa_rms_norm_weighted(b.hidden.as<float>(), m.enorm.f32(),
        b.norm.as<float>(), 2048, count, float(spec_.rms_epsilon), stream_);
    cuda_check(cudaMemcpy2DAsync(m.concat.data(), 4096 * sizeof(float), b.norm.data(),
        2048 * sizeof(float), 2048 * sizeof(float), count, cudaMemcpyDeviceToDevice, stream_), "MTP embedding concat");
    kernels::native_qsa_rms_norm_weighted(normalized_main, m.hnorm.f32(),
        b.norm.as<float>(), 2048, count, float(spec_.rms_epsilon), stream_);
    cuda_check(cudaMemcpy2DAsync(m.concat.as<float>() + 2048, 4096 * sizeof(float), b.norm.data(),
        2048 * sizeof(float), 2048 * sizeof(float), count, cudaMemcpyDeviceToDevice, stream_), "MTP hidden concat");
    if (mtp_sink_) {
        mtp_sink_->stage(0, m.concat.as<float>(), 2048, stream_);
        mtp_sink_->stage(1, m.concat.as<float>() + 2048, 2048, stream_);
    }
    batch_quantize_q8(m.concat.as<float>(), count, 4096);
    batch_project_native(m.eh_proj, b.hidden.as<float>(), count, 4096, 2048, false);
    if (mtp_sink_) mtp_sink_->stage(2, b.hidden.as<float>(), 2048, stream_);
}

// Past draft cells need only K/V. Query, attention, MoE and LM head are omitted
// because none can affect this full-attention decoder's cached keys or values.
void Runtime::mtp_kv_batch(const int* tokens, int count, const float* normalized_main, int first_cell) {
    if (first_cell < 0 || count < 1 || first_cell > kv_capacity_ || count > kv_capacity_ - first_cell)
        throw std::out_of_range("MTP catch-up cells");
    mtp_condition(tokens, count, normalized_main, first_cell + 1);
    auto& layer = mtp_->decoder; auto& b = batch_;
    kernels::native_qsa_rms_norm_weighted(b.hidden.as<float>(), layer.attn_norm.f32(),
        b.norm.as<float>(), 2048, count, float(spec_.rms_epsilon), stream_);
    batch_quantize_q8(b.norm.as<float>(), count, 2048);
    batch_project_native(layer.k, b.k.as<float>(), count, 2048, 512, false);
    batch_project_native(layer.v, b.v.as<float>(), count, 2048, 512, false);
    if (prefill_config_.mtp_kv_bulk && !decode_phase_ && count >= 4) {
        store_attention_kv_rows(layer, b.k.as<float>(), b.v.as<float>(), first_cell, count, 1);
        ++b.mtp_kv_bulk_calls;
    } else for (int t = 0; t < count; ++t)
        store_attention_kv(layer, b.k.as<float>() + size_t(t) * 512,
            b.v.as<float>() + size_t(t) * 512, first_cell + t, 1);
}

int Runtime::mtp_step(int token, int cell, const float* normalized_main, Metrics& metrics) {
    if (!mtp_ || cell < 0 || cell >= kv_capacity_) throw std::out_of_range("MTP draft cell");
    mtp_condition(&token, 1, normalized_main, cell + 1);
    cuda_check(cudaMemcpyAsync(work_.hidden.data(), batch_.hidden.data(), 2048 * sizeof(float),
        cudaMemcpyDeviceToDevice, stream_), "MTP conditioned token");
    auto& m = *mtp_;
    decode_layer(size_t(spec_.n_layers), spec_.mtp[0].decoder, m.decoder, cell, metrics, 1);
    if (mtp_sink_) {
        mtp_sink_->stage(6, work_.hidden.as<float>(), 2048, stream_);
        mtp_sink_->stage(8, batch_.weights.as<float>(), 8, stream_);
    }
    const int next = head_token(metrics, &m.output_norm, m.output.data.bytes() ? &m.output : nullptr);
    if (mtp_sink_) mtp_sink_->stage(7, work_.norm.as<float>(), 2048, stream_);
    return next;
}
} // namespace strata::qwen35moe
