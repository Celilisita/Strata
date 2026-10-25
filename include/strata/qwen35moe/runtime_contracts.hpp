#pragma once
#include "strata/qwen35moe/spec.hpp"
#include "strata/qwen35moe/metrics.hpp"
#include "strata/qwen35moe/device_buffer.hpp"
#include <cuda_runtime.h>
#include <array>
#include <string>
#include <vector>
namespace strata::qwen35moe {
struct GpuWeight {
    DeviceBuffer data;
    int type = -1;
    void load(const GgufFile& gguf, const TensorInfo* tensor);
    const void* ptr() const { return data.data(); }
    const float* f32() const { return data.as<float>(); }
};

struct LayerWeights {
    GpuWeight attn_norm, post_norm, router;
    GpuWeight shared_gate, shared_up, shared_down;
    GpuWeight qkv, gate, alpha, beta, conv, dt, a, ssm_norm, ssm_out;
    GpuWeight q, k, v, out, q_norm, k_norm;
    DeviceBuffer history, state, k_cache, v_cache;
};

struct Workspace {
    DeviceBuffer hidden, norm, mixer, q8_input, q8_wide, q8_small;
    DeviceBuffer qkv, z, alpha, beta, gate, raw, activated, core, post;
    DeviceBuffer qfull, q, k, v, attn, gated, step, status, qpos, kpos;
    DeviceBuffer logits, ids, weights, parts, gu_gate, gu_up, down;
    DeviceBuffer gate_out, up_out, expert_h, shared, ffn;
    DeviceBuffer emb_row, head_logits;
    void allocate(int vocab, size_t embedding_row_bytes);
};


// Optional bounded checkpoints. The runtime never reads files or synchronizes
// for these callbacks; the benchmark owns any sampling and comparison.
struct CheckpointSink {
    virtual ~CheckpointSink() = default;
    virtual void gdn_norm_input(const float*, int, cudaStream_t) {}
    // Optional batch-stage diagnostics; callers supply exact live byte spans.
    virtual void batch_stage(const char*, const void*, size_t, bool, cudaStream_t) {}
    // Normalized/rotated batch queries are ephemeral; diagnostics own any copy.
    virtual void attention_batch_query(int, int, const float*, cudaStream_t) {}
    // Expanded all-row GDN diagnostics use their own benchmark byte budget.
    // Existing sampled sinks intentionally ignore these additional spans.
    virtual void gdn_detail(const char*, const void*, size_t, bool, cudaStream_t) {}
    virtual void layer(int position, size_t layer, const float* values,
                       cudaStream_t stream, Metrics& metrics) = 0;
    virtual void attention(int position, const float* q, const float* k,
                           const float* output, cudaStream_t stream,
                           Metrics& metrics) = 0;
};


struct CpuExpertState;
struct MtpState;
struct SamplingState;
struct SessionSet;
struct SessionState;
struct VisualState;
struct MoeGraphState;
struct MtpCheckpointSink {
    virtual ~MtpCheckpointSink() = default;
    virtual void stage(int index, const float* values, size_t count, cudaStream_t stream) = 0;
};
struct SpeculativeStep {
    int committed = 0, next = -1;
    std::array<int, 2> ids{};
    std::array<float, 2> scores{};
};

struct MtpLayerCheck {
    std::array<double, 9> nrms{};
    int next = -1;
};

struct PinnedExpertLayer {
    uint8_t* data = nullptr;
    size_t stride = 0;
};

struct PinnedExpertSet {
    std::vector<PinnedExpertLayer> layers;
    size_t bytes = 0;
    std::string error;
    PinnedExpertSet() = default;
    PinnedExpertSet(const PinnedExpertSet&) = delete;
    PinnedExpertSet& operator=(const PinnedExpertSet&) = delete;
    ~PinnedExpertSet() {
        for (auto& layer : layers)
            if (layer.data) cudaFreeHost(layer.data);
    }
};

struct BatchProjectionCheck {
    double nrms = 0.0;
    double max_abs = 0.0;
    double native_ms = 0.0;
    double tensor_core_ms = 0.0;
    double grouped_nrms = 0.0;
    double grouped_ms = 0.0;
};

struct PrefillProviderCheck {
    std::array<double, 6> nrms{};
    std::array<float, 2> gdn_ms{};
    int first_route_difference = -1;
    double state_nrms = 0, history_nrms = 0;
};

struct BatchProbe {
    double embed = -1, norm = -1, qkv = -1, z = -1;
    double gdn_final = -1, mixer = -1, native_mixer = -1, f32_mixer = -1;
    double q8_gemm_mixer = -1, oracle_native_mixer = -1;
    double oracle_f16_mixer = -1, oracle_f32_mixer = -1;
    double oracle_q8_gemm_mixer = -1;
    double residual = -1, post_norm = -1;
    double logits = -1, moe_out = -1, shared = -1, layer = -1;
    std::array<int32_t, 8> ids{};
    std::array<float, 8> weights{};
};

} // namespace strata::qwen35moe
