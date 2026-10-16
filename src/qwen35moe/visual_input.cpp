#include "strata/qwen35moe/visual_input.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
namespace strata::qwen35moe {
ImageEmbedding read_image_embedding(const std::string& path) {
    std::ifstream in(path, std::ios::binary); std::array<int32_t, 5> h{};
    if (!in.read(reinterpret_cast<char*>(h.data()), sizeof(h)) || h[0] != 0x31455653 ||
        h[1] < 1 || h[1] > 16384 || h[2] < 1 || h[3] < 1 ||
        int64_t(h[2]) * h[3] != h[1] || h[4] < 1 || h[4] > 16384)
        throw std::invalid_argument("invalid Strata image embedding header");
    ImageEmbedding image{h[2], h[3], h[4], std::vector<float>(size_t(h[1]) * h[4])};
    if (!in.read(reinterpret_cast<char*>(image.rows.data()), std::streamsize(image.rows.size() * sizeof(float))) ||
        in.peek() != std::char_traits<char>::eof() ||
        std::any_of(image.rows.begin(), image.rows.end(), [](float x) { return !std::isfinite(x); }))
        throw std::invalid_argument("invalid Strata image embedding payload");
    return image;
}
VisualInput make_visual_input(const std::vector<int>& tokens, std::vector<ImageEmbedding> images, int image_token) {
    if (tokens.empty() || tokens.size() > 16384 || images.empty() || images.size() > 64)
        throw std::invalid_argument("bounded image prompt required");
    VisualInput result; result.positions.resize(tokens.size());
    size_t image_index = 0; int logical = 0;
    for (size_t cell = 0; cell < tokens.size();) {
        if (tokens[cell] != image_token) { result.positions[cell++] = {logical, logical, logical}; ++logical; continue; }
        if (image_index >= images.size()) throw std::invalid_argument("unmatched image pad");
        auto& image = images[image_index++];
        const int64_t n = int64_t(image.width) * image.height;
        if (image.width < 1 || image.height < 1 || n < 1 || n > 16384 || image.embedding_width < 1 ||
            image.rows.size() != size_t(n) * image.embedding_width || size_t(n) > tokens.size() - cell)
            throw std::invalid_argument("image grid/embedding mismatch");
        const int first = int(cell);
        for (int i = 0; i < n; ++i) {
            if (tokens[cell] != image_token) throw std::invalid_argument("image pad run too short");
            result.positions[cell++] = {logical, logical + i / image.width, logical + i % image.width};
        }
        logical += (std::max)(image.width, image.height);
        result.spans.push_back({first, std::move(image)});
    }
    if (image_index != images.size()) throw std::invalid_argument("unused image embedding");
    return result;
}
} // namespace strata::qwen35moe
