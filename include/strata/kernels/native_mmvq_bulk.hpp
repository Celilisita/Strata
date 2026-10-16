#pragma once
#include <cstddef>

namespace strata::kernels {
// Canonical exact column computation, with all column tiles in grid.y.
// Quant formats, K loop, warp layout and reduction are unchanged. Caller owns
// contiguous Q8_1 input and column-major output; pointers may not overlap.
// No allocation, transfer or synchronization. Shape/pointer rules match MMVQ;
// columns are 8..8192. Tile width is8,16 or32. A remainder uses the canonical 1..7-column dispatch.
bool native_mmvq_bulk_supported(int ggml_type) noexcept;
std::size_t native_mmvq_bulk(int ggml_type, const void* weights, const void* input,
    float* output, int n_in, int n_out, int columns, void* stream, int tile_columns = 8, int tile_rows = 1);
} // namespace strata::kernels
