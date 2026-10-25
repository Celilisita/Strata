#include "strata/gemma4/native/vector_grouped_aligned_q4.hpp"
#include "vector3_layout.cuh"
#include "q4_aligned_reader.cuh"
#include <cstdint>

namespace strata::gemma4::native {
namespace {
using detail::ResidualBlock;
static_assert(sizeof(ResidualBlock) == 120 && offsetof(ResidualBlock, code) == 24);
bool aligned(const void* p) { return p && reinterpret_cast<std::uintptr_t>(p) % 4 == 0; }
bool overlaps(const void* a, size_t na, const void* b, size_t nb) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < na : x - y < nb;
}

__device__ float term(const unsigned char* raw, const ResidualBlock& x,
    int group, float previous) {
    int weight[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) weight[w] = detail245::codes(raw, group, w * 4);
    float product = 0, correction = 0;
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        int first = 0, second = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int code = reinterpret_cast<const int*>(x.code[r])[w];
            if (w < 4) first = __dp4a(weight[w], code, first);
            else second = __dp4a(weight[w], code, second);
        }
        product = __fmaf_rn(x.d[r], float(first + second), product);
        correction = __fmaf_rn(x.d[r], float(x.sum[r]), correction);
    }
    const unsigned meta = detail245::metadata(raw, group);
    const unsigned delta = *reinterpret_cast<const unsigned*>(raw);
    const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(delta))) * float(meta & 63);
    const float minimum = __half2float(__ushort_as_half(static_cast<unsigned short>(delta >> 16))) * float(meta >> 6);
    return __fadd_rn(previous, __fmaf_rn(scale, product, -__fmul_rn(minimum, correction)));
}

__device__ float sum8(float value, unsigned active) {
    for (int distance = 4; distance; distance /= 2)
        value = __fadd_rn(value, __shfl_xor_sync(active, value, distance));
    return value;
}

__global__ void multiply(primitive::Matrix matrix, const ResidualBlock* input, float* output) {
    const int row = int(blockIdx.x) * 16 + threadIdx.x / 8, group = threadIdx.x % 8;
    if (row >= matrix.rows) return;
    const unsigned active = __activemask();
    const int blocks = matrix.columns / 256;
    float value = 0.f;
    for (int k = 0; k < blocks; ++k) {
        const auto* raw = static_cast<const unsigned char*>(matrix.data) + (size_t(row) * blocks + k) * 144;
        value = term(raw, input[k * 8 + group], group, value);
    }
    value = sum8(value, active);
    if (!group) output[row] = value;
}
}

void linear_vector_grouped_aligned_q4_t1(primitive::Matrix matrix, const void* input,
    float* output, int tokens, cudaStream_t stream) {
    if (!aligned(matrix.data) || matrix.layout != primitive::BlockLayout::Gguf ||
        matrix.quant != primitive::Quant::Q4K || matrix.rows < 1 || matrix.rows > 262144 ||
        matrix.columns < 256 || matrix.columns > 21504 || matrix.columns % 256 ||
        !aligned(input) || !aligned(output) || tokens != 1 || !stream)
        throw std::invalid_argument("Gemma T1 aligned Q4 three-residual extent/format");
    const size_t raw_bytes = size_t(matrix.rows) * (matrix.columns / 256) * 144;
    const size_t input_bytes = size_t(matrix.columns / 32) * sizeof(ResidualBlock);
    const size_t output_bytes = size_t(matrix.rows) * sizeof(float);
    if (overlaps(input, input_bytes, output, output_bytes) ||
        overlaps(matrix.data, raw_bytes, output, output_bytes))
        throw std::invalid_argument("Gemma T1 aligned Q4 three-residual output lifetime");
    multiply<<<unsigned((matrix.rows + 15) / 16), 128, 0, stream>>>(
        matrix, static_cast<const ResidualBlock*>(input), output);
    primitive::check_cuda(cudaGetLastError(), "Gemma T1 aligned Q4 three-residual projection");
}
}
