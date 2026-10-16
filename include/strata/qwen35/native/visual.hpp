#pragma once
#include "strata/qwen35/native/weights.hpp"
#include <array>
#include <string>
#include <vector>

namespace strata::qwen35::native {
struct VisualSpan { int first=0, count=0; std::vector<float> rows; const float* device=nullptr; };
struct DenseVisualInput {
    std::vector<int> ids;
    std::vector<std::array<int,3>> positions;
    std::vector<VisualSpan> spans;
    DeviceStorage storage;
    int* table=nullptr;
    int width=0;
    void upload(int context, cudaStream_t);
    void overlay(float* destination, int start, int count, cudaStream_t) const;
};
std::unique_ptr<DenseVisualInput> read_visual_prompt(const std::string&, int context, int width, int vocab);
}
