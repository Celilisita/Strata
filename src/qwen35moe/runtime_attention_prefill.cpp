#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/attention_tile.hpp"
#include "strata/qwen35moe/attention_tc.hpp"
#include "strata/qwen35moe/attention_short_prefill.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::batch_attention_prefill(LayerWeights& layer, int count, int first) {
    auto& b = batch_; auto& w = work_;
    if (first < 0 || first > kv_capacity_ || count > kv_capacity_ - first ||
        attention_scratch_.bytes() < 2 * flash_attn_tile_scratch_bytes(first + count))
        throw std::invalid_argument("unreserved bulk attention prefill");
    // Projection outputs are already canonical. Split all query rows, normalize
    // each independent head, then apply the existing metadata-driven IMRoPE.
    cuda_check(cudaMemcpy2DAsync(b.gated.data(), 256 * sizeof(float), b.qfull.data(),
        512 * sizeof(float), 256 * sizeof(float), size_t(count) * 16,
        cudaMemcpyDeviceToDevice, stream_), "bulk query split");
    kernels::native_qsa_rms_norm_weighted(b.gated.as<float>(), layer.q_norm.f32(),
        b.gated.as<float>(), 256, count * 16, float(spec_.rms_epsilon), stream_);
    attention_prefill_positions(b.attention_qpos.as<int32_t>(), count, 16, first, stream_);
    kernels::native_rope_apply_positions(b.gated.as<float>(), b.gated.as<float>(),
        count * 16, 256, 64, float(spec_.rope_base), b.attention_qpos.as<int>(), stream_, visual_table());
    kernels::native_qsa_rms_norm_weighted(b.k.as<float>(), layer.k_norm.f32(),
        b.k.as<float>(), 256, count * 2, float(spec_.rms_epsilon), stream_);
    attention_prefill_positions(b.attention_kpos.as<int32_t>(), count, 2, first, stream_);
    kernels::native_rope_apply_positions(b.k.as<float>(), b.k.as<float>(), count * 2,
        256, 64, float(spec_.rope_base), b.attention_kpos.as<int>(), stream_, visual_table());
    kernels::f32_to_f16_bulk(b.k.as<float>(), layer.k_cache.as<uint16_t>() + size_t(first) * 512,
        count * 512, stream_);
    kernels::f32_to_f16_bulk(b.v.as<float>(), layer.v_cache.as<uint16_t>() + size_t(first) * 512,
        count * 512, stream_);
    dense_profile_.launches(8);
    dense_profile_.copies(1, 0, uint64_t(count) * 4096 * sizeof(float));
    dense_profile_.mark(DenseStage::Preprocess, stream_);
    if (checkpoint_sink_) checkpoint_sink_->attention_batch_query(first, count, b.gated.as<float>(), stream_);
    // Keep the accepted short-vector arithmetic for the first256 causal cells.
    const int short_rows = (std::max)(0, (std::min)(count, 256 - first));
    if (short_rows && prefill_config_.short_attention_batch) {
        attention_short_prefill(b.gated.as<float>(), layer.k_cache.as<uint16_t>(),
            layer.v_cache.as<uint16_t>(), first, short_rows, kv_capacity_,
            b.post.as<float>(), w.status.as<int32_t>(), stream_);
        ++b.short_attention_calls;
        dense_profile_.launches(1);
    } else {
        for (int t = 0; t < short_rows; ++t) {
            const int position = first + t;
            cuda_check(cudaMemcpyAsync(w.q.data(), b.gated.as<float>() + size_t(t) * 4096,
                4096 * sizeof(float), cudaMemcpyDeviceToDevice, stream_), "short query view");
            const int32_t step[4]{position, position + 1, (position + 1) / 4, position + 1};
            cuda_check(cudaMemcpyAsync(w.step.data(), step, sizeof(step), cudaMemcpyHostToDevice, stream_), "short step");
            attend_cached(layer, position);
            cuda_check(cudaMemcpyAsync(b.post.as<float>() + size_t(t) * 4096, w.attn.data(),
                4096 * sizeof(float), cudaMemcpyDeviceToDevice, stream_), "short attention result");
        }
        dense_profile_.launches(short_rows);
        dense_profile_.copies(uint64_t(short_rows) * 3, uint64_t(short_rows) * 16,
            uint64_t(short_rows) * 2 * 4096 * sizeof(float));
    }
    dense_profile_.mark(DenseStage::ShortAttention, stream_);
    if (short_rows < count && prefill_config_.tensor_attention &&
        attention_tc_fits(first + short_rows, count - short_rows, kv_capacity_)) {
        attention_tc_prefill(b.gated.as<float>() + size_t(short_rows) * 4096,
            layer.k_cache.as<uint16_t>(), layer.v_cache.as<uint16_t>(), first + short_rows,
            count - short_rows, kv_capacity_, b.post.as<float>() + size_t(short_rows) * 4096, stream_, prefill_config_.attention_tc_compact, prefill_config_.attention_tc_scratch_reuse);
        dense_profile_.launches(1);
        ++b.tensor_attention_calls;
        if (prefill_config_.attention_tc_compact) ++b.attention_compact_calls;
        if (prefill_config_.attention_tc_scratch_reuse) ++b.attention_scratch_calls;
    } else if (short_rows < count) {
        if (dense_profile_.active()) throw std::invalid_argument("dense profile requires accepted WMMA core");
        auto* scratch = &attention_scratch_;
        // GDN activation storage is dead in an attention layer. Its phase-arena
        // view is disjoint from Q/query/post; reuse it without growing VRAM.
        if (prefill_config_.attention_queries > 2 && b.scan_activated.bytes() > scratch->bytes())
            scratch = &b.scan_activated;
        const float* packed_q = nullptr;
        if (prefill_config_.packed_attention_query) {
            auto* slots = b.qfull.as<float>() + size_t(short_rows) * 8192;
            attention_pack_query(b.gated.as<float>() + size_t(short_rows) * 4096,
                slots, count - short_rows, stream_);
            packed_q = slots;
        }
        flash_attn_tile_prefill(b.gated.as<float>() + size_t(short_rows) * 4096,
            layer.k_cache.as<uint16_t>(), layer.v_cache.as<uint16_t>(), first + short_rows,
            count - short_rows, kv_capacity_, b.post.as<float>() + size_t(short_rows) * 4096,
            w.status.as<int32_t>(), scratch->as<float>(), scratch->bytes(), stream_,
            prefill_config_.attention_queries, packed_q ? 2 : prefill_config_.attention_group,
            prefill_config_.shared_attention_join, prefill_config_.padded_attention_cache, packed_q, prefill_config_.full_shared_attention);
    }
    dense_profile_.mark(DenseStage::Core, stream_);
    kernels::native_qsa_gate_apply(b.post.as<float>(), b.qfull.as<float>(), b.gated.as<float>(),
        count * 16, 256, stream_);
    dense_profile_.launches(1);
    dense_profile_.mark(DenseStage::Post, stream_);
    batch_quantize_q8(b.gated.as<float>(), count, 4096);
    dense_profile_.mark(DenseStage::OutputQuant, stream_);
    batch_project_native(layer.out, b.mixer.as<float>(), count, 4096, 2048, false);
    dense_profile_.mark(DenseStage::OutputProjection, stream_);
}
} // namespace strata::qwen35moe
