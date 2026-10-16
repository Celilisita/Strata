#include "strata/qwen35moe/runtime.hpp"
#include "strata/kernels/native_qsa.hpp"

#include <array>
#include <chrono>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
constexpr std::array<size_t, 5> kAuxLayers{3, 11, 19, 27, 35};

class AuxSink final : public CheckpointSink {
public:
    AuxSink(int first, int count, std::vector<float>& output)
        : first_(first), count_(count), output_(output) {}

    void layer(int position, size_t layer, const float* values,
               cudaStream_t stream, Metrics&) override {
        for (size_t channel = 0; channel < kAuxLayers.size(); ++channel) {
            if (layer != kAuxLayers[channel]) continue;
            const int row = position - first_;
            if (row < 0 || row >= count_) throw std::out_of_range("DSpark auxiliary row");
            float* target = output_.data() + (size_t(row) * 5 + channel) * 2048;
            cuda_check(cudaMemcpyAsync(target, values, 2048 * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream),
                       "capture verified target hidden state");
            return;
        }
    }

    void attention(int, const float*, const float*, const float*,
                   cudaStream_t, Metrics&) override {}

private:
    int first_, count_;
    std::vector<float>& output_;
};

struct VerifyGuard {
    CheckpointSink*& sink;
    float*& state;
    float*& history;
    CheckpointSink* previous;
    ~VerifyGuard() { sink = previous; state = nullptr; history = nullptr; }
};
}

void Runtime::prepare_dspark_verify(DsparkVerifyState& state, int capacity) {
    if (capacity < 2 || capacity > 16 || mtp_enabled_ || !spec_.mtp.empty())
        throw std::invalid_argument("DSpark requires this MiniCPM target without MTP");
    state.capacity = capacity;
    state.norm.reset(size_t(capacity) * 2048 * sizeof(float));
    state.logits.reset(size_t(capacity) * spec_.n_vocab * sizeof(float));
    state.picks.reset(size_t(capacity) * sizeof(HeadPick));
    state.host_picks.resize(size_t(capacity));
    state.auxiliary.resize(size_t(capacity) * 5 * 2048);
    state.state.resize(layers_.size());
    state.history.resize(layers_.size());
    for (size_t il = 0; il < layers_.size(); ++il) {
        if (!spec_.layers[il].recurrent) continue;
        state.state[il].reset(size_t(capacity - 1) * layers_[il].state.bytes());
        state.history[il].reset(size_t(capacity - 1) * layers_[il].history.bytes());
    }
}

DsparkVerifiedBlock Runtime::verify_dspark_block(DsparkVerifyState& state,
    const std::vector<int>& tokens, int position, Metrics& metrics) {
    const int count = int(tokens.size());
    if (!decode_phase_ || mtp_enabled_ || sampling_ || !gpu_head_ || !prefill_scan_ ||
        count < 1 || count > state.capacity || count > batch_.capacity ||
        position < 0 || position + count > kv_capacity_ ||
        checkpoint_sink_ || state.host_picks.size() < size_t(count))
        throw std::invalid_argument("unprepared DSpark greedy verification block");
    for (int token : tokens)
        if (token < 0 || token >= int(spec_.n_vocab))
            throw std::out_of_range("DSpark proposal outside target vocabulary");
    auto& b = batch_;
    for (int row = 0; row < count; ++row)
        load_embedding(tokens[size_t(row)], position + row,
            b.hidden.as<float>() + size_t(row) * 2048,
            spec_.embedding, work_.emb_row, embedding_row_bytes_);
    AuxSink sink(position, count, state.auxiliary);
    VerifyGuard guard{checkpoint_sink_, verify_state_, verify_history_, checkpoint_sink_};
    checkpoint_sink_ = &sink;
    try {
        for (size_t il = 0; il < layers_.size(); ++il) {
            verify_state_ = count > 1 && spec_.layers[il].recurrent
                ? state.state[il].as<float>() : nullptr;
            verify_history_ = count > 1 && spec_.layers[il].recurrent
                ? state.history[il].as<float>() : nullptr;
            decode_batch_layer(il, count, position, "", metrics);
        }
    } catch (...) {
        cudaStreamSynchronize(stream_);
        throw;
    }
    checkpoint_sink_ = guard.previous;
    verify_state_ = verify_history_ = nullptr;
    kernels::native_qsa_rms_norm_weighted(b.hidden.as<float>(), head_norm_.f32(),
        state.norm.as<float>(), 2048, count, float(spec_.rms_epsilon), stream_);
    batch_quantize_q8(state.norm.as<float>(), count, 2048);
    batch_project_native(head_, state.logits.as<float>(), count, 2048,
                         int(spec_.n_vocab), false);
    for (int row = 0; row < count; ++row)
        select_head_greedy(state.logits.as<float>() + size_t(row) * spec_.n_vocab,
            int(spec_.n_vocab), head_partials_.as<HeadPick>(),
            state.picks.as<HeadPick>() + row, stream_,
            tg_dynamic_ ? &dynamic_frame_.as<core::NativeMissFrame>()->failed : nullptr);
    cuda_check(cudaMemcpyAsync(state.host_picks.data(), state.picks.data(),
        size_t(count) * sizeof(HeadPick), cudaMemcpyDeviceToHost, stream_),
        "observe DSpark target picks");
    const auto observe = std::chrono::steady_clock::now();
    cuda_check(cudaStreamSynchronize(stream_), "complete DSpark block verification");
    metrics.head_wait_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - observe).count();
    metrics.head_d2h += size_t(count) * sizeof(HeadPick);
    ++metrics.explicit_syncs;
    for (int row = 0; row < count; ++row) {
        const auto& pick = state.host_picks[size_t(row)];
        if (pick.first_bad != -1 || pick.id < 0 || pick.id >= int(spec_.n_vocab))
            throw std::runtime_error("invalid DSpark target head or device continuation");
    }
    int committed = 1;
    while (committed < count &&
           state.host_picks[size_t(committed - 1)].id == tokens[size_t(committed)])
        ++committed;
    if (committed < count) {
        const size_t checkpoint = size_t(committed - 1);
        for (size_t il = 0; il < layers_.size(); ++il) {
            if (!spec_.layers[il].recurrent) continue;
            auto& layer = layers_[il];
            cuda_check(cudaMemcpyAsync(layer.state.data(),
                state.state[il].as<uint8_t>() + checkpoint * layer.state.bytes(),
                layer.state.bytes(), cudaMemcpyDeviceToDevice, stream_),
                "restore accepted DSpark recurrent state");
            cuda_check(cudaMemcpyAsync(layer.history.data(),
                state.history[il].as<uint8_t>() + checkpoint * layer.history.bytes(),
                layer.history.bytes(), cudaMemcpyDeviceToDevice, stream_),
                "restore accepted DSpark recurrent history");
            metrics.rollback_bytes += layer.state.bytes() + layer.history.bytes();
        }
        cuda_check(cudaStreamSynchronize(stream_), "complete DSpark recurrent rollback");
        ++metrics.explicit_syncs;
    }
    sampling_history(tokens.data(), committed);
    selected_logits_ = state.logits.as<float>() + size_t(committed - 1) * spec_.n_vocab;
    host_pick_ = state.host_picks[size_t(committed - 1)];
    DsparkVerifiedBlock result;
    result.next = host_pick_.id;
    result.verified = count;
    result.committed_ids.assign(tokens.begin(), tokens.begin() + committed);
    result.auxiliary.assign(state.auxiliary.begin(),
        state.auxiliary.begin() + size_t(committed) * 5 * 2048);
    metrics.verify_tokens += size_t(count);
    collect_dynamic_metrics(metrics);
    return result;
}
} // namespace strata::qwen35moe
