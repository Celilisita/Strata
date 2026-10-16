#pragma once
#include "strata/qwen35moe/runtime.hpp"
namespace strata::qwen35moe {
struct VisualState {
    DeviceBuffer table;
    VisualInput input;
    std::vector<std::array<int32_t, 3>> upload;
    explicit VisualState(int context) : table(size_t(context) * 3 * sizeof(int32_t)), upload(size_t(context)) {}
};
} // namespace strata::qwen35moe
