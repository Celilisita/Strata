#include "strata/qwen35moe/runtime.hpp"

#include <algorithm>

namespace strata::qwen35moe {

void Runtime::reset_session() {
    finish_decode();
    cuda_check(cudaStreamSynchronize(stream_), "finish Qwen35 request before reset");
    if (miss_service_) miss_service_->stop();
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
    batch_.expert_stage_layer = size_t(-1);
    batch_.expert_stage_next = 0;
    batch_.expert_stage_tick = 0;
    batch_.expert_stage_keys.fill(-1);
    std::fill(batch_.expert_stage_slot_keys.begin(), batch_.expert_stage_slot_keys.end(), -1);
    std::fill(batch_.expert_stage_age.begin(), batch_.expert_stage_age.end(), 0);
    reset_sampling();
    clear_visual_input();
    batch_probe_ = BatchProbe{};
    host_pick_.id = -1; selected_logits_ = nullptr;
    decode_phase_ = false;
}

} // namespace strata::qwen35moe
