#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace strata::qwen35moe {
struct ImageEmbedding {
    int width = 0, height = 0, embedding_width = 0;
    std::vector<float> rows;
};
struct VisualSpan { int first = 0; ImageEmbedding image; };
struct VisualInput {
    std::vector<VisualSpan> spans;
    std::vector<std::array<int32_t, 3>> positions;
};
ImageEmbedding read_image_embedding(const std::string& path);
VisualInput make_visual_input(const std::vector<int>& tokens, std::vector<ImageEmbedding> images,
    int image_token);
} // namespace strata::qwen35moe
