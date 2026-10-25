#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::qwen35moe {
inline constexpr int kAttentionMaxContext = 262144;
// Scratch is caller-owned and reused on the model's ordered stream.
std::size_t flash_attn_tile_scratch_bytes(int max_context);
void flash_attn_tile_step(const float* q, const uint16_t* k, const uint16_t* v,
                          const int32_t* step, int capacity, int width, float* output,
                          int32_t* status, const uint16_t* mask, float* scratch,
                          std::size_t scratch_bytes, void* stream);
void attention_prefill_positions(int32_t* out, int count, int heads, int first, void* stream);
// Q/gate slots have512 F32 values per head; only first128 slots are overwritten.
void attention_pack_query(const float* q, float* q_gate_slots, int count, void* stream);
void flash_attn_tile_prefill(const float* q, const uint16_t* k, const uint16_t* v,
    int first, int count, int capacity, float* out, int32_t* status,
    float* scratch, std::size_t scratch_bytes, void* stream, int query_batch = 2, int query_group = 1, bool shared_join = false, bool padded_cache = false, const float* packed_q = nullptr, bool full_shared = false);
}
