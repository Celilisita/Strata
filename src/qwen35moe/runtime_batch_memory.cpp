#include "strata/qwen35moe/runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/qwen35moe/attention_tile.hpp"

#include <algorithm>
#include <stdexcept>

namespace strata::qwen35moe {

// Resize request workspaces only. Weights, residency and providers keep ownership.
void Runtime::allocate_batch(int capacity) {
    auto& b = batch_;
    const size_t t = size_t(capacity);
    const auto f32 = [](size_t n) { return n * sizeof(float); };
    const bool large = capacity > 1024;
    b.phase_arena.reset(large ? f32(t * 28672) : 0);
    auto phase = [&](DeviceBuffer& buffer, size_t elements, size_t first) {
        if (large) buffer.view(b.phase_arena, f32(t * first), f32(elements));
        else buffer.reset(f32(elements));
    };
    b.hidden.reset(f32(t * 2048)); b.norm.reset(f32(t * 2048));
    b.mixer.reset(f32(t * 2048)); phase(b.qkv, t * 8192, 0);
    phase(b.z, t * 4096, 16384); b.alpha.reset(f32(t * 32));
    b.beta.reset(f32(t * 32)); phase(b.post, t * 4096, 24576);
    phase(b.qfull, t * 8192, 0); b.k.reset(f32(t * 512));
    b.v.reset(f32(t * 512)); phase(b.gated, t * 4096, 20480);
    if (capacity >= 1024 && tile_attention_) {
        b.attention_qpos.reset(t * 16 * sizeof(int32_t));
        b.attention_kpos.reset(t * 2 * sizeof(int32_t));
        const size_t partials = 2 * flash_attn_tile_scratch_bytes(kv_capacity_);
        if (attention_scratch_.bytes() < partials) attention_scratch_.reset(partials);
    }
    if (capacity >= 1024 && routed_tc_enabled_ && !b.routed_tc)
        b.routed_tc = std::make_unique<prefill::NativeTcProvider>(stream_);
    b.input_half.reset(t * 4096 * sizeof(uint16_t));
    if (prefill_tc_ && capacity >= 1024) {
        const size_t shared_elements = size_t(spec_.n_embd * spec_.n_ff_shared);
        b.input_low.reset(t * 4096 * sizeof(uint16_t));
        b.weight_f32.reset(shared_elements * sizeof(float));
        b.weight_low.reset(shared_elements * sizeof(uint16_t));
    }
    if (prefill_scan_) {
        phase(b.scan_activated, t * 8192, 8192);
        phase(b.scan_core, t * 4096, 20480);
        b.scan_gate.reset(f32(t * 32));
    }
    b.input_q8.reset(kernels::native_q8_1_bytes(4096) * t);
    b.logits.reset(f32(t * 257)); b.ids.reset(t * 8 * sizeof(int32_t));
    b.weights.reset(f32(t * 8));
    const size_t experts = t * 8;
    if (expert_cache_.slots <= (std::min)(experts, size_t(spec_.n_expert))) {
        const size_t gu = kernels::native_mmvq_weight_bytes(12, 2048, 512);
        const size_t down = kernels::native_mmvq_weight_bytes(14, 512, 2048);
        b.expert_stage_stride = (2 * gu + down + 255) & ~size_t(255);
        size_t free_bytes = 0, total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "budget staged experts");
        const size_t reserve = (std::max)(size_t(512) << 20, free_bytes / 4);
        const size_t affordable = free_bytes > reserve
            ? (free_bytes - reserve) / b.expert_stage_stride : 0;
        b.expert_stage_slots = (std::min)(size_t(256), affordable);
        if (b.expert_stage_slots < (std::min)(experts, size_t(spec_.n_expert)))
            throw std::runtime_error("Qwen35 GPU has no room for grouped expert workset");
        b.expert_stage.reset(b.expert_stage_slots * b.expert_stage_stride);
        b.expert_stage_keys.fill(-1);
        b.expert_stage_slot_keys.assign(b.expert_stage_slots, -1);
        b.expert_stage_age.assign(b.expert_stage_slots, 0);
    }
    b.expert_addresses.reset(experts * 2 * sizeof(unsigned long long));
    b.expert_rows.reset(experts * 2 * sizeof(int32_t));
    b.down_addresses.reset(experts * sizeof(unsigned long long));
    b.down_rows.reset(experts * sizeof(int32_t));
    phase(b.expert_gu, experts * 2 * 512, 0);
    phase(b.expert_hidden, experts * 512, 8192);
    b.expert_q8.reset(kernels::native_q8_1_bytes(512) * experts);
    phase(b.expert_parts, experts * 2048, 0);
    b.expert_mixed_parts.reset(f32(8 * 2048));
    phase(b.shared_gate, t * 512, 16384); phase(b.shared_up, t * 512, 16896);
    phase(b.shared_hidden, t * 512, 17408);
    b.shared_q8.reset(kernels::native_q8_1_bytes(512) * t);
    phase(b.shared_output, t * 2048, 17920);
    b.host_expert_addresses.resize(experts * 2);
    b.host_expert_rows.resize(experts * 2);
    b.host_down_addresses.resize(experts);
    b.host_down_rows.resize(experts);
    b.capacity = capacity;
}

} // namespace strata::qwen35moe
