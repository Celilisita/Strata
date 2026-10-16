#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace strata::qwen35moe {
enum class DenseStage : size_t {
    Input, Query, Key, Value, Preprocess, ShortAttention, Core, Post,
    OutputQuant, OutputProjection, Residual, Count
};
constexpr size_t dense_stages = size_t(DenseStage::Count);
struct DenseCost {
    std::array<double, dense_stages> gpu_ms{}, submit_ms{};
    // Native wrapper launches are exact; opaque MMQ fixups/memsets excluded.
    std::array<uint64_t, dense_stages> launches{};
    std::array<uint64_t, dense_stages> copies{}, h2d_bytes{}, d2d_bytes{};
    uint64_t layers = 0;
};

} // namespace strata::qwen35moe
