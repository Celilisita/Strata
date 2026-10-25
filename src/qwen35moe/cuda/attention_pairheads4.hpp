#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::qwen35moe {
struct AttentionPairHeads4Resources {
    int device, major, minor, multiprocessors, binary_version;
    int threads, registers, max_threads, active_blocks, reason_mask;
    size_t static_shared, local_bytes, shared_limit;
};
// Single fixed CUDA device per host thread, as owned by Runtime. Cached once;
// no per-token resource query, allocation, event, or device synchronization.
AttentionPairHeads4Resources attention_pairheads4_resources();
// Probe-only native observation, with no new gate on the original launcher.
AttentionPairHeads4Resources flash_attn_tile_native_resources();
bool tg_attention_pairheads4_enabled();
// Explicit production A/B entry. Returns actual tile selection. Both choices
// use flash_attn_tile.cu's same canonical join and original buffer validation.
// Unsupported geometry/resources fall back before any candidate device write.
bool flash_attn_tile_step_provider(const float* q, const uint16_t* k, const uint16_t* v,
    const int32_t* step, int capacity, int width, float* output, int32_t* status,
    const uint16_t* mask, float* scratch, size_t scratch_bytes, void* stream,
    bool request_pairheads4);
// Tile-only helper; the provider above owns the canonical join.
bool try_launch_attention_pairheads4(const float* q, const uint16_t* k, const uint16_t* v,
    const int32_t* step, int capacity, int width, int splits,
    const uint16_t* mask, float* partial, int32_t* status, void* stream);
}
