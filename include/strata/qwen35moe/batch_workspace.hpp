#pragma once
#include "strata/qwen35moe/device_buffer.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/native_tc_provider.hpp"
#include "strata/prefill/native_grouped_mmq.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
namespace strata::qwen35moe {
struct BatchWorkspace {
    // Destroy views before their phase arena; all uses follow the model stream.
    DeviceBuffer phase_arena;
    DeviceBuffer hidden, norm, mixer, qkv, z, alpha, beta, post;
    DeviceBuffer qfull, k, v, gated, input_half, input_q8;
    DeviceBuffer attention_qpos, attention_kpos;
    DeviceBuffer input_low, weight_f32, weight_low;
    DeviceBuffer scan_activated, scan_core, scan_gate;
    DeviceBuffer logits, ids, weights;
    DeviceBuffer expert_addresses, expert_rows, down_addresses, down_rows;
    DeviceBuffer expert_stage;
    size_t expert_stage_stride = 0, expert_stage_slots = 0;
    size_t expert_stage_layer = size_t(-1), expert_stage_next = 0;
    uint64_t expert_stage_tick = 0;
    std::array<int32_t, 256> expert_stage_keys{};
    std::vector<int32_t> expert_stage_slot_keys;
    std::vector<uint64_t> expert_stage_age;
    DeviceBuffer expert_gu, expert_hidden, expert_q8, expert_parts;
    DeviceBuffer expert_mixed_parts;
    DeviceBuffer shared_gate, shared_up, shared_hidden, shared_q8, shared_output;
    std::vector<unsigned long long> host_expert_addresses, host_down_addresses;
    std::vector<int32_t> host_expert_rows, host_down_rows;
    std::vector<uint8_t> embedding_rows;
    prefill::Gemm gemm;
    std::unique_ptr<prefill::NativeTcProvider> routed_tc;
    std::unique_ptr<prefill::mmq::NativeGroupedMmqProvider> down_mmq;
    uint64_t gu_active_calls = 0;
    uint64_t mtp_kv_bulk_calls = 0, attention_compact_calls = 0;
    uint64_t router_tiled_calls = 0, qkv_q4_mmq_calls = 0;
    uint64_t gu_mmq_calls = 0;
    uint64_t qkv_mmq_calls = 0;
    uint64_t gdn_out_mmq_calls = 0;
    uint64_t short_attention_calls = 0;
    uint64_t attention_scratch_calls = 0;
    uint64_t down_mmq_calls = 0;
    uint64_t mmq_projection_calls = 0, tensor_attention_calls = 0;
    int capacity = 0;
    int half_width = 0, half_count = 0;
    // Nonowning originalF32 view, rebound by each actualQ8 preparation.
    const float* raw_input = nullptr;
    int raw_input_width = 0, raw_input_count = 0;
    void clear_raw_input() { raw_input=nullptr; raw_input_width=0; raw_input_count=0; }
    size_t owned_bytes() const {
        size_t total = 0;
        for (const auto* buffer : {&phase_arena, &hidden, &norm, &mixer, &qkv, &z, &alpha,
            &beta, &post, &qfull, &k, &v, &gated, &input_half, &input_q8, &attention_qpos,
            &attention_kpos, &input_low, &weight_f32, &weight_low, &scan_activated,
            &scan_core, &scan_gate, &logits, &ids, &weights, &expert_addresses, &expert_rows,
            &down_addresses, &down_rows, &expert_stage, &expert_gu, &expert_hidden,
            &expert_q8, &expert_parts, &expert_mixed_parts, &shared_gate, &shared_up,
            &shared_hidden, &shared_q8, &shared_output}) total += buffer->owned_bytes();
        return total;
    }
};

} // namespace strata::qwen35moe
