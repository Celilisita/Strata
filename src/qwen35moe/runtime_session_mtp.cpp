#include "session_state.hpp"
#include "session_cache_view.hpp"
#include "mtp_state.hpp"
#include "strata/kernels/native_qsa.hpp"

namespace strata::qwen35moe {
// Catch up two independent decoder caches using one conditioning/K/V projection
// schedule. Queries, attention, expert routing and draft heads are omitted.
void Runtime::mtp_kv_sessions(const std::array<int, 2>& tokens, const std::array<int, 2>& positions) {
    auto& m = *mtp_; auto& b = batch_; auto& other = sessions_->slots[1];
    const float* seeds[2]{m.seed.as<float>(), other.mtp_seed.as<float>()};
    for (int t = 0; t < 2; ++t)
        cuda_check(cudaMemcpyAsync(m.main_norm.as<float>() + size_t(t) * 2048, seeds[t],
            2048 * sizeof(float), cudaMemcpyDeviceToDevice, stream_), "gather session MTP seeds");
    mtp_condition(tokens.data(), 2, m.main_norm.as<float>(), positions[0], &positions);
    auto& layer = m.decoder;
    kernels::native_qsa_rms_norm_weighted(b.hidden.as<float>(), layer.attn_norm.f32(),
        b.norm.as<float>(), 2048, 2, float(spec_.rms_epsilon), stream_);
    batch_quantize_q8(b.norm.as<float>(), 2, 2048);
    batch_project_native(layer.k, b.k.as<float>(), 2, 2048, 512, false);
    batch_project_native(layer.v, b.v.as<float>(), 2, 2048, 512, false);
    for (int t = 0; t < 2; ++t) {
        SessionCacheView view(layer.k_cache, layer.v_cache,
            t ? &other.mtp_key : nullptr, t ? &other.mtp_value : nullptr);
        store_attention_kv(layer, b.k.as<float>() + size_t(t) * 512,
            b.v.as<float>() + size_t(t) * 512, positions[size_t(t)] - 1, 1, t);
    }
}
} // namespace strata::qwen35moe
