#include "strata/qwen35moe/runtime.hpp"
#include "head_readback.hpp"

#include <algorithm>

namespace strata::qwen35moe {

void Runtime::reset_session(bool keep_recurrent) {
    finish_decode();
    prefetch_drain();
    cuda_check(cudaStreamSynchronize(stream_), "finish Qwen35 request before reset");
    if (head_readback_) head_readback_->leave_decode();
    batch_.clear_raw_input();
    if (miss_service_) miss_service_->stop();
    if (!keep_recurrent) {
        for (auto& layer : layers_) {
            if (layer.history.data())
                cuda_check(cudaMemsetAsync(layer.history.data(), 0, layer.history.bytes(), stream_),
                           "reset Qwen35 GDN history");
            if (layer.state.data())
                cuda_check(cudaMemsetAsync(layer.state.data(), 0, layer.state.bytes(), stream_),
                           "reset Qwen35 GDN state");
            // Attention reads only positions written by the current request.
            // Clear recurrent state, but do not zero entire long-context K/V arrays.
        }
    }
    batch_.expert_stage_layer = size_t(-1);
    batch_.expert_stage_next = 0;
    batch_.expert_stage_tick = 0;
    batch_.expert_stage_keys.fill(-1);
    std::fill(batch_.expert_stage_slot_keys.begin(), batch_.expert_stage_slot_keys.end(), -1);
    std::fill(batch_.expert_stage_age.begin(), batch_.expert_stage_age.end(), 0);
    if (!keep_recurrent) reset_sampling();
    clear_visual_input();
    batch_probe_ = BatchProbe{};
    host_pick_.id = -1; selected_logits_ = nullptr;
    decode_phase_ = false;
}

void Runtime::save_prefill_checkpoint(int position) {
    if (layers_.empty() || position < 1) return;
    checkpoint_history_.resize(layers_.size());
    checkpoint_state_.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        if (!layer.history.data() || !layer.state.data()) continue;
        auto& history = checkpoint_history_[i];
        auto& state = checkpoint_state_[i];
        if (history.bytes() != layer.history.bytes()) history.reset(layer.history.bytes());
        if (state.bytes() != layer.state.bytes()) state.reset(layer.state.bytes());
        cuda_check(cudaMemcpyAsync(history.data(), layer.history.data(), layer.history.bytes(),
                                   cudaMemcpyDeviceToDevice, stream_), "prefill checkpoint history");
        cuda_check(cudaMemcpyAsync(state.data(), layer.state.data(), layer.state.bytes(),
                                   cudaMemcpyDeviceToDevice, stream_), "prefill checkpoint state");
    }
    prefill_checkpoint_position_ = position;
}

bool Runtime::restore_prefill_checkpoint() {
    if (prefill_checkpoint_position_ < 1 || checkpoint_history_.size() != layers_.size()) return false;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        if (!layer.history.data() || !layer.state.data()) continue;
        if (checkpoint_history_[i].bytes() != layer.history.bytes() ||
            checkpoint_state_[i].bytes() != layer.state.bytes()) return false;
        cuda_check(cudaMemcpyAsync(layer.history.data(), checkpoint_history_[i].data(),
                                   layer.history.bytes(), cudaMemcpyDeviceToDevice, stream_),
                   "restore prefill checkpoint history");
        cuda_check(cudaMemcpyAsync(layer.state.data(), checkpoint_state_[i].data(),
                                   layer.state.bytes(), cudaMemcpyDeviceToDevice, stream_),
                   "restore prefill checkpoint state");
    }
    return true;
}

} // namespace strata::qwen35moe
