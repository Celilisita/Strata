#include "strata/qwen35moe/runtime.hpp"
#include <stdexcept>
namespace strata::qwen35moe {
uint64_t Runtime::batch_moe_gu_mmq(int count, Metrics& metrics) {
    if (decode_phase_ || !prefill_config_.grouped_gu_mmq || count < 1024) return 0;
    auto& b = batch_;
    if (!b.down_mmq) throw std::runtime_error("unprepared grouped GU MMQ provider");
    // The same generic provider/control leases and dead dense phase tail serve
    // GU then Down on the ordered stream. No second expert/activation store.
    void* packed = b.qkv.data(); size_t capacity = b.qkv.bytes();
    if (b.phase_arena.data()) {
        const size_t offset = size_t(b.capacity) * 20480 * sizeof(float);
        if (offset > b.phase_arena.bytes()) throw std::runtime_error("invalid GU phase arena");
        packed = b.phase_arena.as<uint8_t>() + offset; capacity = b.phase_arena.bytes() - offset;
    }
    const int entries = count * 8;
    const auto work = b.down_mmq->run(12, b.host_expert_addresses.data(),
        b.host_expert_addresses.data() + entries, b.host_expert_rows.data(),
        entries, count, b.input_q8.data(), b.expert_addresses.data(), b.expert_addresses.bytes(),
        packed, capacity, b.expert_gu.as<float>(), 2048, 512, prefill_config_.grouped_gu_active);
    metrics.control_h2d += work.control_bytes;
    ++b.gu_mmq_calls;
    if (prefill_config_.grouped_gu_active) ++b.gu_active_calls;
    return uint64_t(work.launches);
}
} // namespace strata::qwen35moe
