#include "session_state.hpp"
#include "sampling_state.hpp"
#include "mtp_state.hpp"
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace strata::qwen35moe {
void Runtime::initialize_sessions() {
    const char* value = std::getenv("STRATA_QWEN35_SESSIONS");
    if (!value || std::string(value) == "1") return;
    if (std::string(value) != "2") throw std::invalid_argument("session capacity must be1 or2");
    if (!gpu_head_) throw std::invalid_argument("sessions require GPU output head");
    auto set = std::make_unique<SessionSet>();
    // Row1 remains a parked sequence view when the active sequence emits a
    // single-token head into row0. No vocabulary D2D copy is needed per pair.
    const size_t single_head_bytes = work_.head_logits.bytes();
    work_.head_logits.reset(2 * single_head_bytes);
    set->additional_bytes += single_head_bytes;
    set->picks.reset(2 * sizeof(HeadPick));
    set->additional_bytes += set->picks.bytes();
    for (auto& slot : set->slots) {
        slot.layers.resize(layers_.size());
        if (sampling_) {
            slot.sampling_tail.reserve(size_t(sampling_->initial.penalty_last_n));
            slot.sampling_counter = sampling_->initial.counter;
        }
    }
    auto& other = set->slots[1];
    auto allocate = [&](DeviceBuffer& dst, const DeviceBuffer& src, bool zero) {
        dst.reset(src.bytes()); set->additional_bytes += src.bytes();
        if (zero && src.bytes()) cuda_check(cudaMemsetAsync(dst.data(), 0, dst.bytes(), stream_), "initialize session state");
    };
    for (size_t il = 0; il < layers_.size(); ++il) {
        auto& dst = other.layers[il]; const auto& src = layers_[il];
        allocate(dst.history, src.history, true); allocate(dst.state, src.state, true);
        allocate(dst.key, src.k_cache, true); allocate(dst.value, src.v_cache, true);
    }
    allocate(other.head_logits, work_.head_logits, false);
    if (visual_) {
        other.visual = std::make_unique<VisualState>(max_context_);
        set->additional_bytes += other.visual->table.bytes();
    }
    if (mtp_) {
        allocate(other.mtp_seed, mtp_->seed, false);
        allocate(other.mtp_key, mtp_->decoder.k_cache, true);
        allocate(other.mtp_value, mtp_->decoder.v_cache, true);
        allocate(other.verify_logits, mtp_->verify_logits, false);
    }
    cuda_check(cudaStreamSynchronize(stream_), "session allocation ready before cache budget");
    sessions_ = std::move(set);
}
int Runtime::session_capacity() const { return sessions_ ? 2 : 1; }
size_t Runtime::additional_session_bytes() const { return sessions_ ? sessions_->additional_bytes : 0; }

void Runtime::select_session(int id) {
    if (id < 0 || id >= session_capacity()) throw std::out_of_range("Qwen35 session ID");
    if (!sessions_ || sessions_->active == id) return;
    auto& set = *sessions_;
    auto& previous = set.slots[size_t(set.active)]; auto& next = set.slots[size_t(id)];
    previous.pick = host_pick_; previous.selected_logits = selected_logits_;
    auto exchange = [&](SessionState& slot) {
        for (size_t il = 0; il < layers_.size(); ++il) {
            auto& state = slot.layers[il]; auto& layer = layers_[il];
            std::swap(state.history, layer.history); std::swap(state.state, layer.state);
            std::swap(state.key, layer.k_cache); std::swap(state.value, layer.v_cache);
        }
        std::swap(slot.head_logits, work_.head_logits);
        if (mtp_) {
            std::swap(slot.mtp_seed, mtp_->seed); std::swap(slot.mtp_key, mtp_->decoder.k_cache);
            std::swap(slot.mtp_value, mtp_->decoder.v_cache); std::swap(slot.verify_logits, mtp_->verify_logits);
        }
        if (sampling_) std::swap(slot.sampling_tail, sampling_->tail);
        std::swap(slot.visual, visual_);
    };
    if (sampling_) previous.sampling_counter = sampling_->current.counter;
    exchange(previous); exchange(next);
    if (sampling_) sampling_->current.counter = next.sampling_counter;
    host_pick_ = next.pick; selected_logits_ = next.selected_logits;
    set.active = id;
    // No GPU copy, allocation, synchronization or decode-epoch restart. Queued
    // kernels retain their pointer values; owners remain alive on the one stream.
}
} // namespace strata::qwen35moe
