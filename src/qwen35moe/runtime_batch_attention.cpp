#include "strata/qwen35moe/runtime.hpp"
#include "session_state.hpp"
#include "session_cache_view.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_rope.hpp"

#include <array>
#include <cstdlib>
#include <stdexcept>

namespace strata::qwen35moe {

void Runtime::batch_attention(LayerWeights& layer, int count, int position_base) {
    auto& b = batch_;
    auto& w = work_;
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    // The persistent KV state requires canonical quantized accumulation.
    batch_project_native(layer.q, f(b.qfull), count, 2048, 8192, false);
    dense_profile_.mark(DenseStage::Query, stream_);
    batch_project_native(layer.k, f(b.k), count, 2048, 512, false);
    dense_profile_.mark(DenseStage::Key, stream_);
    batch_project_native(layer.v, f(b.v), count, 2048, 512, false);
    dense_profile_.mark(DenseStage::Value, stream_);
    const char* bulk = std::getenv("STRATA_QWEN35_BATCH_ATTN");
    if (bulk && std::string(bulk) != "control" && std::string(bulk) != "1")
        throw std::invalid_argument("BATCH_ATTN must be control or1");
    if (bulk && std::string(bulk) == "1" && !decode_phase_ && !paired_session_ &&
        tile_attention_ && count >= 1024) {
        batch_attention_prefill(layer, count, position_base);
        return;
    }
    if (dense_profile_.active()) throw std::invalid_argument("dense profile requires bulk attention");
    for (int t = 0; t < count; ++t) {
        auto* other = paired_session_ && t == 1 ? &paired_session_->layers[paired_layer_] : nullptr;
        SessionCacheView view(layer.k_cache, layer.v_cache, other ? &other->key : nullptr, other ? &other->value : nullptr);
        const int position = paired_session_ ? paired_positions_[size_t(t)] : position_base + t;
        float* qfull = f(b.qfull) + size_t(t) * 8192;
        float* k = f(b.k) + size_t(t) * 512;
        float* v = f(b.v) + size_t(t) * 512;
        cuda_check(cudaMemcpy2DAsync(f(w.q), 256 * sizeof(float), qfull,
                                      512 * sizeof(float), 256 * sizeof(float), 16,
                                      cudaMemcpyDeviceToDevice, stream_), "batch split query");
        kernels::native_qsa_rms_norm_weighted(f(w.q), layer.q_norm.f32(), f(w.q),
                                               256, 16, float(spec_.rms_epsilon), stream_);
        std::array<int32_t, 16> q_positions{};
        q_positions.fill(position);
        cuda_check(cudaMemcpyAsync(w.qpos.data(), q_positions.data(), sizeof(q_positions),
                                    cudaMemcpyHostToDevice, stream_), "batch query position");
        kernels::native_rope_apply_positions(f(w.q), f(w.q), 16, 256, 64,
                                   float(spec_.rope_base), w.qpos.as<int>(), stream_, visual_table(paired_session_ ? t : -1));
        store_attention_kv(layer, k, v, position, 0, paired_session_ ? t : -1);
        const int32_t step[4] = {position, position + 1, (position + 1) / 4, position + 1};
        cuda_check(cudaMemcpyAsync(w.step.data(), step, sizeof(step),
                                    cudaMemcpyHostToDevice, stream_), "batch attention step");
        attend_cached(layer, position);
        kernels::native_qsa_gate_apply(f(w.attn), qfull,
                                       f(b.gated) + size_t(t) * 4096, 16, 256, stream_);
    }
    batch_quantize_q8(f(b.gated), count, 4096);
    batch_project_native(layer.out, f(b.mixer), count, 4096, 2048, false);
}

} // namespace strata::qwen35moe
