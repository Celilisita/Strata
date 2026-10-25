#include "strata/qwen35moe/runtime.hpp"
#include <stdexcept>
#include "strata/qwen35moe/gu_plan_cost.hpp"
#include "strata/prefill/native_code_sum_support.hpp"
#include "runtime_gu_copy_pipeline.hpp"
namespace strata::qwen35moe {
uint64_t Runtime::batch_moe_gu_mmq(int count, Metrics& metrics) {
    if (decode_phase_ || !prefill_config_.grouped_gu_mmq || count < 1024) return 0;
    if (!prefill::mmq::native_code_sum_grouped_supported(12, 2048, 512)) return 0;
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
    prefill::mmq::NativeGroupedMmqWork work;
    try{work = gu_copy_pending_ ? b.down_mmq->run_gu_pipeline(b.host_expert_addresses.data(),
        b.host_expert_addresses.data()+entries,b.host_expert_rows.data(),entries,count,b.input_q8.data(),
        b.expert_addresses.data(),b.expert_addresses.bytes(),packed,capacity,b.expert_gu.as<float>(),
        prefill_config_.gu_copy_wave,[&](const prefill::mmq::NativeMmqMatrix* matrices,int n,bool last){
            gu_copy_->ready(prefetch_cache(),matrices,n,last,stream_);
        }) : b.down_mmq->run(12, b.host_expert_addresses.data(),
        b.host_expert_addresses.data() + entries, b.host_expert_rows.data(),
        entries, count, b.input_q8.data(), b.expert_addresses.data(), b.expert_addresses.bytes(),
        packed, capacity, b.expert_gu.as<float>(), 2048, 512, prefill_config_.grouped_gu_active);
    }catch(...){if(gu_copy_pending_)batch_failed_=true;throw;}
    if(gu_copy_pending_){const auto& copied=gu_copy_->cost();
        metrics.expert_h2d+=copied.bytes;metrics.expert_h2d_calls+=copied.calls;
        metrics.expert_h2d_logical_copies+=copied.logical_copies;
        metrics.expert_h2d_registered_bytes+=copied.registered_bytes;
        metrics.expert_h2d_pageable_bytes+=copied.pageable_bytes;
        metrics.expert_cache_misses+=copied.cold;metrics.expert_enqueue_ms+=copied.enqueue_ms;
    }
    record_gu_plan_cost(metrics, work, prefill_config_.grouped_gu_active);
    metrics.control_h2d += work.control_bytes;
    ++b.gu_mmq_calls;
    if (prefill_config_.grouped_gu_active) ++b.gu_active_calls;
    return uint64_t(work.launches);
}
} // namespace strata::qwen35moe
