#include "strata/qwen35moe/cuda_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
struct alignas(4) Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

__global__ void reconstruct(const Q81Block* __restrict__ blocks,
                            uint16_t* __restrict__ output, uint16_t* __restrict__ residual, int elements) {
    const int i = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    if (i >= elements) return;
    const Q81Block& block = blocks[i / 32];
    const float d = __half2float(__low2half(block.ds));
    const float value = float(block.qs[i % 32]) * d;
    const half high = __float2half_rn(value);
    output[i] = __half_as_ushort(high);
    if (residual) residual[i] = __half_as_ushort(
        __float2half_rn((value - __half2float(high)) * 4096.0f));
}
}

void q8_1_to_f16(const void* blocks, uint16_t* output, int elements, void* stream, uint16_t* residual) {
    if (elements <= 0 || elements % 32 || !blocks || !output || !stream)
        throw std::invalid_argument("invalid Qwen35 Q8_1 reconstruction");
    reconstruct<<<(elements + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(blocks), output, residual, elements);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::qwen35moe
