#include "strata/qwen35moe/runtime.hpp"
#include "mtp_state.hpp"
#include "strata/kernels/native_qsa.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::mtp_prompt(const int* tokens, int count, const float* hidden, Metrics& metrics) {
    auto& m = *mtp_;
    // head_token already produced the last main normalized row. Preserve it
    // before conditioning reuses scratch; the seed is never a pre-norm residual.
    cuda_check(cudaMemcpyAsync(m.seed.data(), work_.norm.data(), 2048 * sizeof(float),
        cudaMemcpyDeviceToDevice, stream_), "MTP prompt final seed");
    const int catchup = (std::min)(batch_.capacity, int(m.main_norm.bytes() / (2048 * sizeof(float))));
    for (int first = 0; first < count - 1; first += catchup) {
        const int n = (std::min)(catchup, count - 1 - first);
        // RMS normalization is row-local; consume each bounded slice before
        // reusing the same scratch on the ordered model stream.
        kernels::native_qsa_rms_norm_weighted(hidden + size_t(first) * 2048,
            head_norm_.f32(), m.main_norm.as<float>(), 2048, n,
            float(spec_.rms_epsilon), stream_);
        mtp_kv_batch(tokens + first + 1, n, m.main_norm.as<float>(), first);
        metrics.kv_catchup_rows += size_t(n);
    }
    cuda_check(cudaStreamSynchronize(stream_), "MTP prompt catch-up ready");
    ++metrics.explicit_syncs;
}

SpeculativeStep Runtime::run_speculative(int token, int position, int remaining,
    const std::vector<int>& stops, Metrics& metrics, bool force_reject) {
    if (!mtp_enabled_ || !mtp_ || !tg_dynamic_ || !decode_phase_ || !gpu_head_ || !prefill_scan_ || batch_.capacity < 2)
        throw std::runtime_error("MTP verify requires prepared canonical scan, dynamic shared cache and GPU head");
    if (token < 0 || token >= int(spec_.n_vocab) || position < 1 || remaining < 1 ||
        position >= kv_capacity_ || (remaining >= 2 && position + 1 >= kv_capacity_))
        throw std::out_of_range("MTP generation window");
    auto& m = *mtp_;
    SpeculativeStep result;
    result.ids[0] = token; result.scores[0] = host_pick_.logit;
    auto copy = [&](void* dst, const void* src, size_t bytes) {
        cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream_), "speculative state copy");
    };
    auto single = [&](bool kv_ready) {
        if (!kv_ready) { mtp_kv_batch(&token, 1, m.seed.as<float>(), position - 1); ++metrics.kv_catchup_rows; }
        result.next = run_token(token, position, true, metrics);
        copy(m.seed.data(), work_.norm.data(), 2048 * sizeof(float));
        result.committed = 1;
    };
    // A two-input tail cannot amortize a rejected draft/verify window. Keep
    // the canonical single-token schedule; KV catch-up preserves continuation.
    if (remaining <= 2) { single(false); return result; }
    if (sampling_) sampling_begin_draft(token);
    const int draft = mtp_step(token, position - 1, m.seed.as<float>(), metrics);
    if (sampling_) sampling_end_draft();
    ++metrics.mtp_windows;
    collect_dynamic_metrics(metrics);
    if (std::find(stops.begin(), stops.end(), draft) != stops.end()) {
        single(true); return result;
    }
    const int verify_tokens[2]{token, draft};
    std::array<HeadPick, 2> picks{};
    verify_pair(verify_tokens, position, picks, metrics);
    const bool accepted = !force_reject && picks[0].id == draft;
    const int final_row = accepted ? 1 : 0;
    if (accepted) {
        result.ids[1] = draft; result.scores[1] = picks[0].logit;
        result.committed = 2; ++metrics.mtp_accepted;
        // Complete the accepted pair(h_main_x,draft_y). The draft already stored
        // the previous pair(h_main_prev,x); no FFN/head is needed for this catch-up.
        mtp_kv_batch(&draft, 1, m.main_norm.as<float>(), position);
        ++metrics.kv_catchup_rows;
    } else {
        result.committed = 1;
        if (sampling_) sampling_reject_verify();
        for (size_t il = 0; il < layers_.size(); ++il) {
            if (!spec_.layers[il].recurrent) continue;
            copy(layers_[il].state.data(), m.prefix_state[il].data(), layers_[il].state.bytes());
            copy(layers_[il].history.data(), m.prefix_history[il].data(), layers_[il].history.bytes());
            metrics.rollback_bytes += layers_[il].state.bytes() + layers_[il].history.bytes();
        }
        // Dense KV has no recurrence: the discarded future cell is excluded by
        // causal length and overwritten at the next committed physical position.
    }
    copy(m.seed.data(), m.main_norm.as<float>() + size_t(final_row) * 2048, 2048 * sizeof(float));
    selected_logits_ = m.verify_logits.as<float>() + size_t(final_row) * spec_.n_vocab;
    host_pick_ = picks[size_t(final_row)]; result.next = host_pick_.id;
    return result;
}
} // namespace strata::qwen35moe
