#include "strata/models/deepseek_v4/hyper_connections.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

void check_dim(int dim) {
    require(dim > 0 && dim <= 16384, "DeepSeek mHC dimension outside supported range");
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

float inverse_rms(std::span<const float> values, float epsilon) {
    require(epsilon > 0 && std::isfinite(epsilon), "invalid mHC norm epsilon");
    float sum = 0.0f;
    for (float value : values) sum += value * value;
    return 1.0f / std::sqrt(sum / static_cast<float>(values.size()) + epsilon);
}

float project(std::span<const float> row, std::span<const float> x) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) sum += row[i] * x[i];
    return sum;
}

void normalize_columns(std::array<float, 16>& matrix, float epsilon) {
    for (int column = 0; column < hc_streams; ++column) {
        float sum = 0.0f;
        for (int row = 0; row < hc_streams; ++row) sum += matrix[row * hc_streams + column];
        for (int row = 0; row < hc_streams; ++row) matrix[row * hc_streams + column] /= sum + epsilon;
    }
}

void normalize_rows(std::array<float, 16>& matrix, float epsilon) {
    for (int row = 0; row < hc_streams; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < hc_streams; ++column) sum += matrix[row * hc_streams + column];
        for (int column = 0; column < hc_streams; ++column) matrix[row * hc_streams + column] /= sum + epsilon;
    }
}

} // namespace

float round_to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    // Keep a NaN a NaN rather than rounding a small payload to infinity.
    if ((bits & 0x7fffffffU) > 0x7f800000U) return std::bit_cast<float>(bits | 0x00400000U);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

void hc_pre(const HcWeights& weights, std::span<const float> residual,
            HcMix& mix, std::span<float> reduced) {
    check_dim(weights.dim);
    const std::size_t width = static_cast<std::size_t>(weights.dim) * hc_streams;
    require(residual.size() == width && reduced.size() == static_cast<std::size_t>(weights.dim),
            "invalid mHC activation shapes");
    require(weights.projection.size() == hc_mix_rows * width && weights.scale.size() == 3 &&
            weights.base.size() == hc_mix_rows, "invalid mHC weight shapes");
    require(weights.sinkhorn_iterations >= 1 && weights.sinkhorn_iterations <= 100 &&
            weights.sinkhorn_epsilon > 0 && std::isfinite(weights.sinkhorn_epsilon),
            "invalid mHC Sinkhorn parameters");
    const float rsqrt = inverse_rms(residual, weights.norm_epsilon);
    std::array<float, hc_mix_rows> projected{};
    for (int row = 0; row < hc_mix_rows; ++row) {
        projected[row] = project(weights.projection.subspan(row * width, width), residual) * rsqrt;
    }
    const float epsilon = weights.sinkhorn_epsilon;
    for (int stream = 0; stream < hc_streams; ++stream) {
        mix.pre[stream] = sigmoid(projected[stream] * weights.scale[0] + weights.base[stream]) + epsilon;
        mix.post[stream] = 2.0f * sigmoid(projected[stream + hc_streams] * weights.scale[1] +
                                        weights.base[stream + hc_streams]);
    }
    auto& combination = mix.combination;
    for (int row = 0; row < hc_streams; ++row) {
        float maximum = -INFINITY;
        for (int column = 0; column < hc_streams; ++column) {
            const int index = row * hc_streams + column;
            combination[index] = projected[index + 2 * hc_streams] * weights.scale[2] +
                                 weights.base[index + 2 * hc_streams];
            maximum = std::max(maximum, combination[index]);
        }
        float sum = 0.0f;
        for (int column = 0; column < hc_streams; ++column) {
            float& value = combination[row * hc_streams + column];
            value = std::exp(value - maximum);
            sum += value;
        }
        for (int column = 0; column < hc_streams; ++column) {
            combination[row * hc_streams + column] = combination[row * hc_streams + column] / sum + epsilon;
        }
    }
    // Official order: initial row softmax + eps, column division, then 19 row/column pairs.
    normalize_columns(combination, epsilon);
    for (int iteration = 1; iteration < weights.sinkhorn_iterations; ++iteration) {
        normalize_rows(combination, epsilon);
        normalize_columns(combination, epsilon);
    }
    for (int channel = 0; channel < weights.dim; ++channel) {
        float sum = 0.0f;
        for (int stream = 0; stream < hc_streams; ++stream) {
            sum += mix.pre[stream] * residual[stream * weights.dim + channel];
        }
        reduced[channel] = round_to_bf16(sum);
    }
}

void hc_post(const HcMix& mix, std::span<const float> residual,
             std::span<const float> sublayer, std::span<float> output) {
    require(!sublayer.empty() && sublayer.size() <= 16384 &&
            residual.size() == hc_streams * sublayer.size() && output.size() == residual.size(),
            "invalid mHC post shapes");
    const std::size_t dim = sublayer.size();
    for (int destination = 0; destination < hc_streams; ++destination) {
        for (std::size_t channel = 0; channel < dim; ++channel) {
            float mixed_residual = 0.0f;
            for (int source = 0; source < hc_streams; ++source) {
                mixed_residual += mix.combination[source * hc_streams + destination] * residual[source * dim + channel];
            }
            output[destination * dim + channel] = round_to_bf16(mix.post[destination] * sublayer[channel] + mixed_residual);
        }
    }
}

void hc_head(int dim, std::span<const float> projection, float scale,
             std::span<const float> base, float norm_epsilon, float hc_epsilon,
             std::span<const float> residual, std::span<float> output) {
    check_dim(dim);
    const std::size_t width = static_cast<std::size_t>(dim) * hc_streams;
    require(projection.size() == hc_streams * width && base.size() == hc_streams &&
            residual.size() == width && output.size() == static_cast<std::size_t>(dim),
            "invalid mHC head shapes");
    require(hc_epsilon > 0 && std::isfinite(hc_epsilon), "invalid mHC head epsilon");
    const float rsqrt = inverse_rms(residual, norm_epsilon);
    std::array<float, hc_streams> pre{};
    for (int stream = 0; stream < hc_streams; ++stream) {
        pre[stream] = sigmoid(project(projection.subspan(stream * width, width), residual) * rsqrt * scale + base[stream]) + hc_epsilon;
    }
    for (int channel = 0; channel < dim; ++channel) {
        float sum = 0.0f;
        for (int stream = 0; stream < hc_streams; ++stream) sum += pre[stream] * residual[stream * dim + channel];
        output[channel] = round_to_bf16(sum);
    }
}

} // namespace strata::deepseek_v4
