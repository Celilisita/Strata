#pragma once
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace strata::qwen35::native {
// Original GGUF Q8_0: one FP16 delta and32 signed codes per block32.
// This weight codec is distinct from the runtime's padded activation blocks.
__device__ __forceinline__ float q8_raw_value(const void *data, int columns, int row, int column) {
    const auto *block = static_cast<const unsigned char *>(data) +
        (size_t(row) * (columns / 32) + column / 32) * 34;
    const float delta = __half2float(*reinterpret_cast<const half *>(block));
    return delta * float(reinterpret_cast<const int8_t *>(block + 2)[column % 32]);
}
} // namespace strata::qwen35::native
