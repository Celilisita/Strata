#include "strata/gemma4/native/model.hpp"
#include "strata/gemma4/native/vector_single.hpp"
#include "strata/gemma4/native/vector_two_plane.hpp"

namespace strata::gemma4::native {
void Model::select_aligned_q6_head(bool enabled) {
    if (executable_ || host_position_ != 0)
        throw std::invalid_argument("Gemma aligned head policy requires reset-to-idle before Graph capture");
    constexpr unsigned head_bits = (1u << 4) | (1u << 9) | (1u << 14);
    if (enabled && (!(projection_q8_mask_ & head_bits) ||
        spec_.output != spec_.embedding || head_.data != embedding_.data))
        throw std::invalid_argument("Gemma aligned head needs explicit single-Q8 mask and tied inventory");
    head_sidecar_.select(enabled, head_, stream_);
}
void Model::project_head(const float* input, float* output, int tokens, bool t1_decode) {
    // PP computes one final head row too; only an actual T1 decoder forward
    // may select the independent two-plane candidate. Verify retains parent.
    if(t1_decode&&tokens==1&&(t1_paths_&4u)){
        const auto arena=ffn_->attention_workspace();
        const TwoPlaneQ8Binding binding{this,++t1_arena_epoch_,stream_};
        const auto encoded=prepare_two_plane_q8_t1(input,arena.data,arena.bytes,spec_.hidden,binding);
        linear_two_plane_q8_t1(head_,encoded,output,binding);++t1_head_enqueues_;
    } else if (single_projection(4, tokens)) {
        const auto encoded = ffn_->prepare_linear_single(input, spec_.hidden, tokens, stream_);
        if (head_sidecar_.enabled()) head_sidecar_.project(encoded, output, stream_);
        else ffn_->linear_single(head_, encoded, output, stream_);
    } else ffn_->linear(head_, input, output, tokens, stream_, false);
}
}
