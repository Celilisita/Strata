#pragma once
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
// Scheduling adapter for contiguous Q8_1 blocks, using the canonical quantizer.
// A block never crosses a row because width is a multiple of32. The unmasked
// quantizer ignores row_width, so flattening rows changes only launch count.
inline size_t native_quantize_rows(const float* input, void* output, int width,
    int rows, bool bulk, void* stream, const unsigned long long* addresses = nullptr) {
    if (width < 32 || width % 32 || rows < 1 ||
        int64_t(width) * rows > (std::numeric_limits<int>::max)())
        throw std::invalid_argument("invalid contiguous Q8_1 rows");
    if (bulk && !addresses) {
        native_quantize_q8_1(input, output, width * rows, 1, stream);
        return 1;
    }
    const size_t stride = native_q8_1_bytes(width);
    size_t launches = 0;
    for (int first = 0; first < rows; first += 8) {
        native_quantize_q8_1(input + size_t(first) * width,
            static_cast<uint8_t*>(output) + size_t(first) * stride,
            width, (std::min)(8, rows - first), stream, addresses ? addresses + first : nullptr);
        ++launches;
    }
    return launches;
}
} // namespace strata::kernels
