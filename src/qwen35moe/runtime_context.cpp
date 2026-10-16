#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/attention_tile.hpp"
#include "mtp_state.hpp"
#include "session_state.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::reserve_context(int tokens, Metrics& metrics, int prefill_capacity) {
    if (tokens < 1 || tokens > max_context_)
        throw std::out_of_range("request context exceeds configured capacity");
    if (prefill_capacity < 0 || prefill_capacity > 8192)
        throw std::invalid_argument("invalid reserved prefill capacity");
    const int desired = (std::max)(prefill_capacity, (std::max)(1024, batch_.capacity));
    if (max_context_ <= 16384 && desired == prefill_budget_capacity_) return;
    // MTP owns a second K/V pair per parked session. Keep its prior
    // single-session reservation contract until both MTP pairs can grow.
    if (decode_phase_ || host_pick_.id != -1 || (sessions_ && mtp_))
        throw std::invalid_argument("context reservation requires a reset request without MTP sessions");
    int target = max_context_ <= 16384 ? kv_capacity_ : 256;
    while (target < tokens && target < max_context_) target = (std::min)(2 * target, max_context_);
    if (sessions_) target = (std::max)(target, kv_capacity_);
    // Reuse high-water K/V when all experts already fit. On constrained devices
    // smaller requests may return K/V space to the same expert directory.
    if (target < kv_capacity_ && (kv_capacity_ <= 16384 ||
        expert_cache_.slots == expert_sources_.size() * 256)) target = kv_capacity_;
    const bool resize_kv = target != kv_capacity_;
    if (!resize_kv && desired == prefill_budget_capacity_) return;
    size_t old_context = 0, new_context = 0;
    auto plan = [&](const LayerWeights& layer, size_t parked_bytes) {
        if (!layer.k_cache.bytes()) return;
        old_context += layer.k_cache.bytes() + layer.v_cache.bytes() + parked_bytes;
        new_context += size_t(target) * 512 * sizeof(uint16_t) * 2 * (sessions_ ? 2 : 1);
    };
    if (resize_kv) {
        for (size_t il = 0; il < layers_.size(); ++il) {
            const auto* parked = sessions_ ? &sessions_->slots[size_t(sessions_->active ^ 1)].layers[il] : nullptr;
            plan(layers_[il], parked ? parked->key.bytes() + parked->value.bytes() : 0);
        }
        if (mtp_) plan(mtp_->decoder, 0);
        old_context += attention_scratch_.owned_bytes();
        if (tile_attention_) new_context += flash_attn_tile_scratch_bytes(target) * 2;
    }
    // Validate before releasing any allocation. Budget includes existing expert
    // backing exactly once, and reserves the future request workspace.
    const size_t slots = enable_expert_cache_ ? expert_cache_budget(
        expert_cache_.data.bytes() + old_context, new_context, target, desired) : 0;
    if (slots && slots < size_t(resident_layers_) * 256 + 8)
        throw std::invalid_argument("context budget cannot preserve resident expert layers");
    cuda_check(cudaStreamSynchronize(stream_), "drain context reservation");
    batch_failed_ = true;
    // Shrinking compacts retained quantized bytes on GPU, then frees only tail
    // chunks. A fitting cache keeps every address, slot, age and its clock.
    if (slots && slots < expert_cache_.slots) resize_expert_cache(slots, metrics);
    auto allocate = [&](LayerWeights& layer, SessionLayerState* parked) {
        if (!layer.k_cache.bytes()) return;
        if (parked) {
            sessions_->additional_bytes -= parked->key.bytes() + parked->value.bytes();
            parked->key.reset(size_t(target) * 512 * sizeof(uint16_t));
            parked->value.reset(size_t(target) * 512 * sizeof(uint16_t));
            sessions_->additional_bytes += parked->key.bytes() + parked->value.bytes();
        }
        layer.k_cache.reset(size_t(target) * 512 * sizeof(uint16_t));
        layer.v_cache.reset(size_t(target) * 512 * sizeof(uint16_t));
    };
    if (resize_kv) {
        for (size_t il = 0; il < layers_.size(); ++il)
            allocate(layers_[il], sessions_ ? &sessions_->slots[size_t(sessions_->active ^ 1)].layers[il] : nullptr);
        if (mtp_) allocate(mtp_->decoder, nullptr);
        if (tile_attention_) attention_scratch_.reset(flash_attn_tile_scratch_bytes(target) * 2);
        kv_capacity_ = target;
    }
    if (slots && slots > expert_cache_.slots) resize_expert_cache(slots, metrics);
    prefill_budget_capacity_ = desired;
    // MoE graph arguments reference the stable address table and are refreshed
    // at start_decode; only prepare_batch growth needs new graph captures.
    batch_failed_ = false;
}
} // namespace strata::qwen35moe
