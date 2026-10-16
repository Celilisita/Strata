#include "strata/qwen35moe/runtime.hpp"
#include "mtp_state.hpp"
#include "sampling_state.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_qsa.hpp"
#include <stdexcept>
#include <chrono>

namespace strata::qwen35moe {
static_assert(offsetof(kernels::SamplePick, id) == offsetof(HeadPick, id));
static_assert(offsetof(kernels::SamplePick, first_bad) == offsetof(HeadPick, first_bad));
static_assert(offsetof(kernels::SamplePick, logit) == offsetof(HeadPick, logit));
namespace {
struct PrefixClear {
    float*& state; float*& history;
    ~PrefixClear() { state = nullptr; history = nullptr; }
};
}
void Runtime::verify_pair(const int* tokens, int position, std::array<HeadPick, 2>& picks, Metrics& metrics) {
    auto& b = batch_; auto& m = *mtp_;
    PrefixClear guard{verify_state_, verify_history_};
    for (int t = 0; t < 2; ++t) {
        load_embedding(tokens[t], position + t, b.hidden.as<float>() + size_t(t) * 2048,
            spec_.embedding, work_.emb_row, embedding_row_bytes_);
    }
    for (size_t il = 0; il < layers_.size(); ++il) {
        verify_state_ = m.prefix_state[il].as<float>();
        verify_history_ = m.prefix_history[il].as<float>();
        decode_batch_layer(il, 2, position, "", metrics);
    }
    verify_state_ = verify_history_ = nullptr;
    kernels::native_qsa_rms_norm_weighted(b.hidden.as<float>(), head_norm_.f32(),
        m.main_norm.as<float>(), 2048, 2, float(spec_.rms_epsilon), stream_);
    batch_quantize_q8(m.main_norm.as<float>(), 2, 2048);
    batch_project_native(head_, m.verify_logits.as<float>(), 2, 2048, int(spec_.n_vocab), false);
    for (int t = 0; t < 2; ++t) {
        if (sampling_) {
            if (t == 1) sampling_verify_prefix();
            sampling_history(tokens + t, 1);
            selected_logits_ = m.verify_logits.as<float>() + size_t(t) * spec_.n_vocab;
            launch_sample(metrics, t);
            // SamplePick's leading12B share HeadPick layout; no vocabulary copy.
            cuda_check(cudaMemcpyAsync(m.picks.as<HeadPick>() + t, sampling_->gpu.result(),
                sizeof(HeadPick), cudaMemcpyDeviceToDevice, stream_), "verify sampled pick");
        } else select_head_greedy(m.verify_logits.as<float>() + size_t(t) * spec_.n_vocab,
            int(spec_.n_vocab), head_partials_.as<HeadPick>(), m.picks.as<HeadPick>() + t,
            stream_, &dynamic_frame_.as<core::NativeMissFrame>()->failed);
    }
    const auto observe_start = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpyAsync(picks.data(), m.picks.data(), sizeof(picks),
        cudaMemcpyDeviceToHost, stream_), "verify head picks");
    const auto before = std::chrono::steady_clock::now();
    cuda_check(cudaStreamSynchronize(stream_), "verify head ready");
    metrics.head_wait_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - before).count();
    metrics.head_observe_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - observe_start).count();
    metrics.head_d2h += sizeof(picks); ++metrics.explicit_syncs;
    for (const auto& pick : picks)
        if (pick.first_bad != -1 || pick.id < 0 || pick.id >= int(spec_.n_vocab))
            throw std::runtime_error("invalid speculative verify head or device continuation failure");
    metrics.verify_tokens += 2;
    collect_dynamic_metrics(metrics);
}
} // namespace strata::qwen35moe
