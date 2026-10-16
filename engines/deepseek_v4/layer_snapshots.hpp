#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
struct ggml_tensor;
namespace strata::deepseek_v4 {
inline constexpr size_t layer_snapshot_layers = 92, layer_snapshot_values = 32768;
inline constexpr const char* layer_snapshot_stages[]{
    "hc_attn_pre", "attn_norm", "qr_norm", "q", "kv_norm", "attn_derope", "attn_wo_a",
    "attn_out", "hc_attn_post", "hc_ffn_pre", "ffn_norm", "ffn_moe_logits", "ffn_moe_topk",
    "ffn_moe_weights_scaled", "ffn_moe_gate", "ffn_moe_up", "ffn_moe_swiglu_limited",
    "ffn_moe_down", "ffn_moe_out", "ffn_shexp", "ffn_out"
};
inline constexpr int layer_snapshot_stage_axes[]{1,1,1,2,2,2,1,1,2,1,1,1,1,2,2,2,2,2,1,1,1};
inline std::string layer_snapshot_name(size_t index) {
    const char* heads[]{"hc_head_mixes", "hc_head_pre", "hc_head", "result_norm"};
    const char* extras[]{"attn_hca-37", "attn_csa_lid-38", "lid_top_k-38"};
    if (index < 43) return "layer" + std::to_string(index);
    if (index < 47) return heads[index-43];
    if (index < 89) return std::string(layer_snapshot_stages[(index-47)%21]) + "-" + std::to_string(37+(index-47)/21);
    return index < 92 ? extras[index-89] : "invalid";
}
inline int layer_snapshot_axis(size_t index) {
    if (index < 43) return 2;
    if (index < 47 || index >= 89) return 1;
    return layer_snapshot_stage_axes[(index-47)%21];
}
inline size_t layer_snapshot_width(size_t index) {
    constexpr size_t stages[]{4096,4096,1024,32768,512,32768,8192,4096,16384,4096,4096,
        256,6,6,12288,12288,12288,24576,4096,4096,4096};
    constexpr size_t heads[]{4,4,4096,4096};
    if (index < 43) return 16384;
    if (index < 47) return heads[index-43];
    if (index < 89) return stages[(index-47)%21];
    return index == 91 ? 512 : 32768;
}
inline bool layer_snapshot_integer(size_t index) { return index == 91 || (index >= 47 && index < 89 && (index-47)%21 == 12); }
inline int layer_snapshot_index(const char* name) noexcept {
    if (std::strcmp(name, "hc_head_mixes") == 0) return 43;
    if (std::strcmp(name, "hc_head_pre") == 0) return 44;
    if (std::strcmp(name, "hc_head") == 0) return 45;
    if (std::strcmp(name, "result_norm") == 0) return 46;
    if (std::strcmp(name, "h_nextn") == 0) return 42;
    for (int layer = 37; layer <= 38; ++layer) {
        const char* suffix = layer == 37 ? "-37" : "-38";
        for (int point = 0; point < 21; ++point) {
            const size_t length = std::strlen(layer_snapshot_stages[point]);
            if (std::strncmp(name, layer_snapshot_stages[point], length) == 0 &&
                std::strcmp(name + length, suffix) == 0) return 47 + (layer-37)*21 + point;
        }
    }
    if (std::strcmp(name, "attn_hca-37") == 0) return 89;
    if (std::strcmp(name, "attn_csa_lid-38") == 0) return 90;
    if (std::strcmp(name, "lid_top_k-38") == 0) return 91;
    if (std::strncmp(name, "l_last-", 7) != 0) return -1;
    const char* cursor = name + 7;
    int layer = 0, digits = 0;
    while (*cursor >= '0' && *cursor <= '9') {
        if (++digits > 2) return -1;
        layer = layer * 10 + (*cursor++ - '0');
    }
    return digits && !*cursor && layer < 43 ? layer : -1;
}
enum class LayerSnapshotAction { Initialize, Begin, End, Read, Release };
struct LayerSnapshotPacket {
    LayerSnapshotAction action;
    void* handle = nullptr;
    float* rows = nullptr;
    uint32_t* observed = nullptr;
    int32_t* types = nullptr;
    size_t* row_values = nullptr;
    size_t values = 0;
    uint64_t mask[2]{~uint64_t(0), (uint64_t(1)<<28)-1};
    int error = 0;
};
using LayerSnapshotProc = bool (*)(LayerSnapshotPacket*);
bool layer_snapshot_api(LayerSnapshotPacket*);
void layer_snapshot_node(const ggml_tensor*, uintptr_t stream) noexcept;
}
