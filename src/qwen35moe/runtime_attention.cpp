#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/qwen35moe/attention_tile.hpp"
#include "cuda/attention_pairheads4.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"

#include <array>
#include <stdexcept>

namespace strata::qwen35moe {

void Runtime::attention_step(LayerWeights& layer, int position, int rope_offset) {
    auto& w = work_;
    auto f = [&](DeviceBuffer& buffer) { return buffer.as<float>(); };
    kernels::native_quantize_q8_1(f(w.norm), w.q8_input.data(), 2048, 1, stream_);
    project(layer.q, w.q8_input.data(), f(w.qfull), 2048, 8192);
    project(layer.k, w.q8_input.data(), f(w.k), 2048, 512);
    project(layer.v, w.q8_input.data(), f(w.v), 2048, 512);
    cuda_check(cudaMemcpy2DAsync(f(w.q), 256 * sizeof(float), f(w.qfull),
                                  512 * sizeof(float), 256 * sizeof(float), 16,
                                  cudaMemcpyDeviceToDevice, stream_), "split query and gate");
    kernels::native_qsa_rms_norm_weighted(f(w.q), layer.q_norm.f32(), f(w.q),
                                           256, 16, float(spec_.rms_epsilon), stream_);
    std::array<int32_t, 16> q_positions{};
    q_positions.fill(position + rope_offset);
    cuda_check(cudaMemcpyAsync(w.qpos.data(), q_positions.data(), sizeof(q_positions),
                                cudaMemcpyHostToDevice, stream_), "query position");
    kernels::native_rope_apply_positions(f(w.q), f(w.q), 16, 256, 64,
                               float(spec_.rope_base), w.qpos.as<int>(), stream_, visual_table());
    store_attention_kv(layer, f(w.k), f(w.v), position, rope_offset);
    const int32_t step[4] = {position, position + 1, (position + 1) / 4, position + 1};
    cuda_check(cudaMemcpyAsync(w.step.data(), step, sizeof(step), cudaMemcpyHostToDevice, stream_),
               "attention step");
    attend_cached(layer, position);
    kernels::native_qsa_gate_apply(f(w.attn), f(w.qfull), f(w.gated), 16, 256, stream_);
    kernels::native_quantize_q8_1(f(w.gated), w.q8_wide.data(), 4096, 1, stream_);
    project(layer.out, w.q8_wide.data(), f(w.mixer), 4096, 2048);
}

// Shared by normal attention and MTP catch-up; one canonical K normalization,
// rotary transform, F16 conversion and physical-cell addressing.
void Runtime::store_attention_kv(LayerWeights& layer, float* k, const float* v,
                                 int position, int rope_offset, int session) {
    if (position < 0 || position >= kv_capacity_)
        throw std::out_of_range("unreserved attention cell");
    kernels::native_qsa_rms_norm_weighted(k, layer.k_norm.f32(), k,
        256, 2, float(spec_.rms_epsilon), stream_);
    std::array<int32_t, 2> positions{};
    positions.fill(position + rope_offset);
    cuda_check(cudaMemcpyAsync(work_.kpos.data(), positions.data(), sizeof(positions),
        cudaMemcpyHostToDevice, stream_), "key position");
    kernels::native_rope_apply_positions(k, k, 2, 256, 64, float(spec_.rope_base), work_.kpos.as<int>(), stream_, visual_table(session));
    const size_t offset = size_t(position) * 512;
    kernels::f32_to_f16_bulk(k, layer.k_cache.as<uint16_t>() + offset, 512, stream_);
    kernels::f32_to_f16_bulk(v, layer.v_cache.as<uint16_t>() + offset, 512, stream_);
}

void Runtime::attend_cached(LayerWeights& layer, int position) {
    auto& w = work_;
    if (tile_attention_ && position >= 256)
        (void)flash_attn_tile_step_provider(w.q.as<float>(), layer.k_cache.as<uint16_t>(),
            layer.v_cache.as<uint16_t>(), w.step.as<int32_t>(), kv_capacity_, position + 1,
            w.attn.as<float>(), w.status.as<int32_t>(), nullptr,
            attention_scratch_.as<float>(), attention_scratch_.bytes(), stream_,
            decode_phase_ && !mtp_enabled_ && position + 1 >= 8192 && tg_attention_pairheads4_enabled());
    else
        flash_attn_short_step(w.q.as<float>(), layer.k_cache.as<uint16_t>(),
            layer.v_cache.as<uint16_t>(), w.step.as<int32_t>(), kv_capacity_, kv_capacity_ < 1024 ? kv_capacity_ : 1024,
            position + 1, w.attn.as<float>(), w.status.as<int32_t>(), nullptr, stream_);
}

} // namespace strata::qwen35moe
