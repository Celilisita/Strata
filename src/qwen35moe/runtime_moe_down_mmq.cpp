#include "strata/qwen35moe/runtime.hpp"
#include <stdexcept>
namespace strata::qwen35moe {
uint64_t Runtime::batch_moe_down_mmq(size_t il, int count, Metrics& metrics) {
    const int type = int(expert_sources_[il].down_exps->type);
    if (decode_phase_ || !prefill_config_.grouped_down_mmq || count < 1024 ||
        (type != 12 && type != 14)) return 0;
    auto& b = batch_;
    if (!b.down_mmq) throw std::runtime_error("unprepared grouped Down MMQ provider");
    // Dense query/post storage is dead until the next mixer. Its arena tail
    // is disjoint from routed output/activation and shared-expert views.
    void* packed = b.qkv.data();
    size_t packed_bytes = b.qkv.bytes();
    if (b.phase_arena.data()) {
        const size_t offset = size_t(b.capacity) * 20480 * sizeof(float);
        if (offset > b.phase_arena.bytes()) throw std::runtime_error("invalid MoE phase arena");
        packed = b.phase_arena.as<uint8_t>() + offset;
        packed_bytes = b.phase_arena.bytes() - offset;
    }
    const int entries = count * 8;
    const auto work = b.down_mmq->run(type, b.host_down_addresses.data(), nullptr,
        b.host_down_rows.data(), entries, entries, b.expert_q8.data(),
        b.expert_addresses.data(), b.expert_addresses.bytes(), packed, packed_bytes,
        b.expert_parts.as<float>(), 512, 2048);
    metrics.control_h2d += work.control_bytes;
    ++b.down_mmq_calls;
    return uint64_t(work.launches);
}
} // namespace strata::qwen35moe
