#pragma once

#include <array>
#include <span>

namespace strata::deepseek_v4 {

// Vision Exp's four-stream mHC, independent of Qwen's gated-residual implementation.
// Inputs contain BF16 values represented as float. All weights and mixing math are F32.
// No expert execution, ownership, allocation, or synchronization occurs here.
inline constexpr int hc_streams = 4;
inline constexpr int hc_mix_rows = 24;

struct HcWeights {
    int dim = 0;
    std::span<const float> projection;  // [24, 4 * dim], row-major
    std::span<const float> scale;       // [3]
    std::span<const float> base;        // [24]
    int sinkhorn_iterations = 20;
    float norm_epsilon = 1e-20f;
    float sinkhorn_epsilon = 1e-6f;
};

struct HcMix {
    std::array<float, hc_streams> pre{};
    std::array<float, hc_streams> post{};
    // [source_stream, destination_stream]; hc_post sums the SOURCE dimension.
    std::array<float, hc_streams * hc_streams> combination{};
};

float round_to_bf16(float value) noexcept;

// Computes normalized F32 projection, Sinkhorn mixing, then BF16-rounded reduction.
// residual: [4, dim]; reduced: [dim]. Buffers must not overlap.
void hc_pre(const HcWeights& weights, std::span<const float> residual,
            HcMix& mix, std::span<float> reduced);

// sublayer: [dim], residual: [4, dim], output: [4, dim].
// Applies post * sublayer + combination^T * residual, then rounds to BF16.
// output must not overlap either input.
void hc_post(const HcMix& mix, std::span<const float> residual,
             std::span<const float> sublayer, std::span<float> output);

// Final mHC head has [4, 4 * dim] projection, ONE scale, and FOUR base values.
// It uses sigmoid + hc_epsilon, with no Sinkhorn or post expansion.
void hc_head(int dim, std::span<const float> projection, float scale,
             std::span<const float> base, float norm_epsilon, float hc_epsilon,
             std::span<const float> residual, std::span<float> output);

} // namespace strata::deepseek_v4
