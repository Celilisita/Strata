#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/attention_tile.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/elementwise.hpp"
#include <stdexcept>
namespace strata::qwen35moe {
// The scalar store_attention_kv expressions, scheduled by independent head rows.
// No attention query, expert computation, staging or allocation is introduced.
void Runtime::store_attention_kv_rows(LayerWeights& layer, float* k, const float* v,
    int first, int count, int rotary_offset) {
    if (!k || !v || first < 0 || count < 4 || count > batch_.capacity ||
        first > kv_capacity_ - count || rotary_offset < 0 || rotary_offset > 1 ||
        batch_.attention_kpos.bytes() < size_t(count) * 2 * sizeof(int32_t))
        throw std::invalid_argument("unreserved batched MTP key/value cells");
    kernels::native_qsa_rms_norm_weighted(k, layer.k_norm.f32(), k, 256, count * 2,
        float(spec_.rms_epsilon), stream_);
    attention_prefill_positions(batch_.attention_kpos.as<int32_t>(), count, 2,
        first + rotary_offset, stream_);
    kernels::native_rope_apply_positions(k, k, count * 2, 256, 64, float(spec_.rope_base),
        batch_.attention_kpos.as<int>(), stream_, visual_table());
    const size_t offset = size_t(first) * 512;
    kernels::f32_to_f16_bulk(k, layer.k_cache.as<uint16_t>() + offset, count * 512, stream_);
    kernels::f32_to_f16_bulk(v, layer.v_cache.as<uint16_t>() + offset, count * 512, stream_);
}
} // namespace strata::qwen35moe
