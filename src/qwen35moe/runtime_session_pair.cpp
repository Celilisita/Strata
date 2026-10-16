#include "session_state.hpp"
#include "sampling_state.hpp"
#include "mtp_state.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_qsa.hpp"
#include <chrono>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::begin_session_pair(const std::array<int, 2>& positions) {
    select_session(0);
    paired_session_ = &sessions_->slots[1]; paired_positions_ = positions;
}
void Runtime::end_session_pair() { paired_session_ = nullptr; }

std::array<int, 2> Runtime::run_session_pair(const std::array<int, 2>& tokens,
    const std::array<int, 2>& positions, Metrics& metrics) {
    if (!sessions_ || !decode_phase_ || !tg_dynamic_ || batch_.capacity < 2 || !prefill_scan_)
        throw std::invalid_argument("session pair requires prepared canonical scan and shared dynamic cache");
    for (int t = 0; t < 2; ++t)
        if (tokens[size_t(t)] < 0 || tokens[size_t(t)] >= int(spec_.n_vocab) ||
            positions[size_t(t)] < 1 || positions[size_t(t)] >= max_context_)
            throw std::out_of_range("session pair token/position");
    select_session(0);
    if (mtp_enabled_) { mtp_kv_sessions(tokens, positions); metrics.kv_catchup_rows += 2; }
    auto& b = batch_;
    for (int t = 0; t < 2; ++t) {
        load_embedding(tokens[size_t(t)], positions[size_t(t)], b.hidden.as<float>() + size_t(t) * 2048,
            spec_.embedding, work_.emb_row, embedding_row_bytes_, t);
    }
    begin_session_pair(positions);
    try {
        for (size_t il = 0; il < layers_.size(); ++il) decode_batch_layer(il, 2, positions[0], "", metrics);
    } catch (...) { end_session_pair(); throw; }
    end_session_pair();
    kernels::native_qsa_rms_norm_weighted(b.hidden.as<float>(), head_norm_.f32(),
        b.norm.as<float>(), 2048, 2, float(spec_.rms_epsilon), stream_);
    if (mtp_enabled_) {
        float* seeds[2]{mtp_->seed.as<float>(), sessions_->slots[1].mtp_seed.as<float>()};
        for (int t = 0; t < 2; ++t) cuda_check(cudaMemcpyAsync(seeds[t], b.norm.as<float>() + size_t(t) * 2048,
            2048 * sizeof(float), cudaMemcpyDeviceToDevice, stream_), "session pair main seed");
    }
    batch_quantize_q8(b.norm.as<float>(), 2, 2048);
    float* logits = work_.head_logits.as<float>();
    batch_project_native(head_, logits, 2, 2048, int(spec_.n_vocab), false);
    for (int t = 0; t < 2; ++t) {
        select_session(t);
        selected_logits_ = logits + size_t(t) * spec_.n_vocab;
        if (sampling_) {
            sampling_history(tokens.data() + t, 1); launch_sample(metrics, t);
            cuda_check(cudaMemcpyAsync(sessions_->picks.as<HeadPick>() + t, sampling_->gpu.result(),
                sizeof(HeadPick), cudaMemcpyDeviceToDevice, stream_), "session sampled pick");
        } else select_head_greedy(selected_logits_, int(spec_.n_vocab), head_partials_.as<HeadPick>(),
            sessions_->picks.as<HeadPick>() + t, stream_, &dynamic_frame_.as<core::NativeMissFrame>()->failed);
    }
    std::array<HeadPick, 2> picks{};
    cuda_check(cudaMemcpyAsync(picks.data(), sessions_->picks.data(), sizeof(picks),
        cudaMemcpyDeviceToHost, stream_), "session pair picks");
    const auto wait = std::chrono::steady_clock::now();
    cuda_check(cudaStreamSynchronize(stream_), "session pair ready");
    metrics.head_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait).count();
    metrics.head_d2h += sizeof(picks); ++metrics.explicit_syncs;
    for (const auto& pick : picks)
        if (pick.first_bad != -1 || pick.id < 0 || pick.id >= int(spec_.n_vocab))
            throw std::runtime_error("invalid session pair head or device continuation failure");
    select_session(0); host_pick_ = picks[0]; selected_logits_ = logits;
    auto& other = sessions_->slots[1]; other.pick = picks[1]; other.selected_logits = logits + spec_.n_vocab;
    collect_visual_metrics(metrics);
    collect_dynamic_metrics(metrics);
    return {picks[0].id, picks[1].id};
}
} // namespace strata::qwen35moe
