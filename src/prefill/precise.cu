#include "strata/prefill/precise.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::prefill {
namespace {
__global__ void split(const float* source, uint16_t* high, uint16_t* low, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const half h = __float2half_rn(source[i]);
    high[i] = __half_as_ushort(h);
    low[i] = __half_as_ushort(__float2half_rn((source[i] - __half2float(h)) * 4096.0f));
}
}
bool precise_dense_candidate(int type) noexcept { return type == 14; }
void precise_native(Gemm& gemm, const uint16_t* input_high, const uint16_t* input_low,
                    int type, const void* weights, float* output,
                    int64_t tokens, int64_t rows, int64_t cols,
                    float* weight_f32, uint16_t* weight_low, int64_t scratch_elements) {
    if (!input_high || !input_low || !weight_f32 || !weight_low || tokens < 1 ||
        rows < 1 || cols < 1 || scratch_elements < cols || gemm.scratch_elems() < cols)
        throw std::invalid_argument("invalid precise native GEMM workspace");
    const int64_t capacity = scratch_elements < gemm.scratch_elems()
        ? scratch_elements : gemm.scratch_elems();
    const int64_t chunk_rows = capacity / cols;
    for (int64_t first = 0; first < rows; first += chunk_rows) {
        const int64_t n = rows - first < chunk_rows ? rows - first : chunk_rows;
        kernels::dequant_f32(type, weights, first, n, cols, weight_f32, gemm.stream());
        split<<<unsigned((n * cols + 255) / 256), 256, 0,
                static_cast<cudaStream_t>(gemm.stream())>>>(
            weight_f32, gemm.scratch(), weight_low, n * cols);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        // Keep the accepted 1024-token Tensor Core accumulation schedule. The
        // decoded weight slice is reused across all token tiles of the ubatch.
        constexpr int64_t token_tile = 1024;
        for (int64_t token = 0; token < tokens; token += token_tile) {
            const int64_t t = tokens - token < token_tile ? tokens - token : token_tile;
            auto* destination = output + token * rows + first;
            gemm.f16(input_high + token * cols, gemm.scratch(), destination, t, n, cols, rows);
            gemm.f16(input_low + token * cols, gemm.scratch(), destination,
                     t, n, cols, rows, 1, 1.0f / 4096);
            gemm.f16(input_high + token * cols, weight_low, destination,
                     t, n, cols, rows, 1, 1.0f / 4096);
        }
    }
}
}
