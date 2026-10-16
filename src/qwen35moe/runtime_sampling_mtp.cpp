#include "sampling_state.hpp"

namespace strata::qwen35moe {
// Couple the draft draw to the next target draw. Drafting never consumes target
// history or randomness: restore both before canonical verification.
void Runtime::sampling_begin_draft(int token) {
    auto& s = *sampling_;
    s.saved_tail = s.tail; s.saved_counter = s.current.counter;
    sampling_history(&token, 1);
}
void Runtime::sampling_end_draft() {
    auto& s = *sampling_;
    s.tail.swap(s.saved_tail); s.current.counter = s.saved_counter;
}
void Runtime::sampling_verify_prefix() { sampling_->saved_tail = sampling_->tail; }
void Runtime::sampling_reject_verify() {
    auto& s = *sampling_;
    s.tail.swap(s.saved_tail);
    --s.current.counter; // Discard the uncommitted second target draw.
}
} // namespace strata::qwen35moe
