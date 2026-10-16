#pragma once
#include <cstddef>
#include <cstdint>
struct ggml_tensor;
namespace strata::deepseek_v4 {
inline constexpr size_t attention_repeat_values = 512 * 64;
inline constexpr size_t attention_repeat_row_bytes = attention_repeat_values * sizeof(float);
inline constexpr size_t attention_repeat_k_capacity = 16ull << 20;
inline constexpr size_t attention_repeat_mask_capacity = 32768;
inline constexpr int attention_repeat_count = 16;
inline constexpr size_t attention_repeat_k_offset = attention_repeat_row_bytes;
inline constexpr size_t attention_repeat_mask_offset = attention_repeat_k_offset + attention_repeat_k_capacity;
inline constexpr size_t attention_repeat_sink_offset = attention_repeat_mask_offset + attention_repeat_mask_capacity;
inline constexpr size_t attention_repeat_output_offset = attention_repeat_sink_offset + 256;
inline constexpr size_t attention_repeat_bytes = attention_repeat_output_offset + (attention_repeat_count+1)*attention_repeat_row_bytes;
enum class AttentionRepeatAction { Initialize, Begin, End, Read, Release };
struct AttentionRepeatPacket {
    AttentionRepeatAction action;
    void* handle = nullptr;
    void* bytes = nullptr;
    size_t capacity = 0, k_bytes = 0, mask_bytes = 0;
    uint32_t observed = 0;
    int error = 0;
};
using AttentionRepeatProc = bool (*)(AttentionRepeatPacket*);
bool attention_repeat_api(AttentionRepeatPacket*);
bool attention_repeat_capture(const ggml_tensor*, uintptr_t stream) noexcept;
void attention_repeat_output(const ggml_tensor*, int sample, uintptr_t stream) noexcept;
}
