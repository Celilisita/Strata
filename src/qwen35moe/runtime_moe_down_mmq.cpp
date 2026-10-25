#include "strata/qwen35moe/runtime.hpp"
#include <stdexcept>
#include "strata/prefill/native_code_sum_support.hpp"
#include "../prefill/native_ordinary_grouped_mmq.hpp"
#ifdef STRATA_BLACKWELL_MMQ
#include "strata/prefill/native_blackwell_mmq.hpp"
#endif
namespace strata::qwen35moe {
uint64_t Runtime::batch_moe_down_mmq(size_t il, int count, Metrics& metrics) {
    const int type = int(expert_sources_[il].down_exps->type);
    bool q5 = false;
#ifdef STRATA_BLACKWELL_MMQ
    q5 = type==13 && prefill::mmq::blackwell_mmq_backend(type,1)==6;
#endif
    if (decode_phase_ || !prefill_config_.grouped_down_mmq || count < 1024 ||
        (type != 12 && type != 13 && type != 14)) return 0;
    // Unsupported Q5 falls back before plan/repack/lease or scratch mutation.
    if (type == 13) {
        if (!q5 && !prefill::mmq::native_q5_down_ordinary_eligible()) return 0;
    } else if (!prefill::mmq::native_code_sum_grouped_supported(type, 512, 2048)) return 0;
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
uint64_t Runtime::batch_moe_down_fused(size_t il, int count, Metrics& metrics) {
#ifndef STRATA_BLACKWELL_MMQ
    (void)il; (void)count; (void)metrics;
    return 0;
#else
    if (!prefill_config_.down_input_fused || decode_phase_ || checkpoint_sink_ ||
        !prefill_config_.grouped_down_mmq || count<1024) return 0;
    const int type=int(expert_sources_[il].down_exps->type);
    if ((type!=12 && type!=13) || prefill::mmq::blackwell_mmq_backend(type,1)!=6) return 0;
    auto& b=batch_;
    if (!b.down_mmq || count>b.capacity || !b.phase_arena.data())
        throw std::runtime_error("unprepared fused Down provider/arena");
    const size_t offset=size_t(b.capacity)*20480*sizeof(float);
    if (offset>b.phase_arena.bytes()) throw std::runtime_error("invalid fused Down arena");
    const int entries=count*8;
    // Existing event5 now ends GU before the fused activation encoder. Total
    // MoE interval remains identical; no new event, stream or drain is added.
    if (!prefill_events_.empty())
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_+il][5],stream_),"prefill GU fused end");
    const auto work=b.down_mmq->run_swiglu_down(type,b.host_down_addresses.data(),
        b.host_down_rows.data(),entries,entries,b.expert_gu.as<float>(),
        b.expert_gu.as<float>()+size_t(entries)*512,b.expert_addresses.data(),
        b.expert_addresses.bytes(),b.phase_arena.as<uint8_t>()+offset,
        b.phase_arena.bytes()-offset,b.expert_parts.as<float>(),512,2048);
    metrics.control_h2d+=work.control_bytes;
    ++b.down_mmq_calls;
    return uint64_t(work.launches);
#endif
}
} // namespace strata::qwen35moe
