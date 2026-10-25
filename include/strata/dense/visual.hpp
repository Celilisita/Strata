#pragma once
#include "strata/dense/spec.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace strata::dense {
struct InputBatch {
    std::vector<int> ids;
    std::vector<int> positions; // Optional section-major t,y,x,z.
    std::vector<int64_t> image_rows; // Local physical row indices.
    std::vector<float> embeddings;
    int next_logical = -1;
};
struct VisualInput {
    int width = 0, next_logical = 0;
    std::vector<int> ids, positions;
    std::vector<int64_t> image_rows;
    std::vector<float> embeddings;
    InputBatch slice(int begin, int end) const;
};
// DVI1 is an internal prepared input, created from a real SDV1 encoder result.
// The caller measures image encoding and preparation with decoder request time.
VisualInput read_visual_input(const std::string &path, const Spec &, int context, int max_new);
} // namespace strata::dense
