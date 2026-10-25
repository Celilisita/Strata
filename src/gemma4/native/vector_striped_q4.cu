#include "strata/gemma4/native/vector_striped_q4.hpp"
#include "small_prefill_values.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::gemma4::native {
namespace {
constexpr int Threads = 128, Rows = 4, Stripes = 4;
bool overlaps(const void* a, size_t na, const void* b, size_t nb) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < na : x - y < nb;
}
bool aligned(const void* p) { return p && reinterpret_cast<std::uintptr_t>(p) % 4 == 0; }

__device__ unsigned metadata(const unsigned char* raw, int group) {
    const unsigned shift = unsigned(group & 3) * 8;
    const unsigned scale_byte = (*reinterpret_cast<const unsigned*>(raw + 4) >> shift) & 255u;
    const unsigned minimum_byte = (*reinterpret_cast<const unsigned*>(raw + 8) >> shift) & 255u;
    if (group < 4) return (scale_byte & 63u) | ((minimum_byte & 63u) << 6);
    const unsigned high = (*reinterpret_cast<const unsigned*>(raw + 12) >> shift) & 255u;
    const unsigned scale = (high & 15u) | ((scale_byte >> 6) << 4);
    const unsigned minimum = (high >> 4) | ((minimum_byte >> 6) << 4);
    return scale | (minimum << 6);
}

// Preserve 202's group32 integer dot, precomputed activation sum, and F32 term.
__device__ float term(const unsigned char* raw, const SingleQ8Block& x,
    int group, float previous) {
    int first = 0, second = 0;
    #pragma unroll
    for (int word = 0; word < 8; ++word) {
        const int weight = primitive::detail::weight_codes<primitive::Quant::Q4K, true>(raw, group, word * 4);
        const int code = reinterpret_cast<const int*>(x.code)[word];
        if (word < 4) first = __dp4a(weight, code, first);
        else second = __dp4a(weight, code, second);
    }
    const unsigned descriptor = metadata(raw, group);
    const unsigned delta = *reinterpret_cast<const unsigned*>(raw);
    const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(delta))) * float(descriptor & 63);
    const float minimum = __half2float(__ushort_as_half(static_cast<unsigned short>(delta >> 16))) * float(descriptor >> 6);
    const float product = __fmaf_rn(x.d, float(first + second), 0.f);
    const float correction = __fmaf_rn(x.d, float(x.sum), 0.f);
    return __fadd_rn(previous, __fmaf_rn(scale, product, -__fmul_rn(minimum, correction)));
}

__device__ float sum_stripes(float value, unsigned active) {
    // First combine eight groups within each K stripe, then the four stripes.
    #pragma unroll
    for (int distance = 4; distance; distance /= 2)
        value = __fadd_rn(value, __shfl_xor_sync(active, value, distance));
    value = __fadd_rn(value, __shfl_xor_sync(active, value, 16));
    return __fadd_rn(value, __shfl_xor_sync(active, value, 8));
}

template<bool Pair>
__global__ void multiply(primitive::Matrix first, primitive::Matrix second,
    const SingleQ8Block* input, float* gate, float* up, float* activation) {
    const int row = int(blockIdx.x) * Rows + threadIdx.x / 32;
    if (row >= first.rows) return; // Only whole warps return; no CTA barrier.
    const int lane = threadIdx.x % 32, group = lane % 8, stripe = lane / 8;
    const unsigned active = __activemask();
    const int blocks = first.columns / 256;
    float a = 0.f, b = 0.f;
    for (int k = stripe; k < blocks; k += Stripes) {
        const size_t offset = (size_t(row) * blocks + k) * 144;
        const auto& code = input[k * 8 + group];
        a = term(static_cast<const unsigned char*>(first.data) + offset, code, group, a);
        if constexpr (Pair)
            b = term(static_cast<const unsigned char*>(second.data) + offset, code, group, b);
    }
    const float value = sum_stripes(a, active);
    float other = 0.f;
    if constexpr (Pair) other = sum_stripes(b, active);
    if (!lane) {
        gate[row] = value;
        if constexpr (Pair) {
            up[row] = other;
            const float z = 0.7978845608028654f * (value + 0.044715f * value * value * value);
            activation[row] = (0.5f * value * (1.f + tanhf(z))) * other;
        }
    }
}

void validate(primitive::Matrix w, SingleQ8View x, cudaStream_t stream) {
    if (!aligned(w.data) || w.quant != primitive::Quant::Q4K ||
        w.layout != primitive::BlockLayout::Gguf || w.rows < 1 || w.rows > 262144 ||
        w.columns < 256 || w.columns > 21504 || w.columns % 256 ||
        !aligned(x.data) || x.columns != w.columns || x.tokens != 1 ||
        x.bytes < size_t(x.columns / 32) * sizeof(SingleQ8Block) || !stream)
        throw std::invalid_argument("Gemma T1 striped Q4 Single40 typed lease");
}

template<bool Pair>
void launch(primitive::Matrix a, primitive::Matrix b, SingleQ8View x,
    float* gate, float* up, float* activation, cudaStream_t stream) {
    multiply<Pair><<<unsigned((a.rows + Rows - 1) / Rows), Threads, 0, stream>>>(
        a, b, x.data, gate, up, activation);
    primitive::check_cuda(cudaGetLastError(), "Gemma T1 striped Q4 projection");
}
}

void linear_single_q4_striped_t1(primitive::Matrix w, SingleQ8View x,
    float* output, cudaStream_t stream) {
    validate(w, x, stream);
    const size_t bytes = size_t(w.rows) * sizeof(float), raw = size_t(w.rows) * (w.columns / 256) * 144;
    if (!aligned(output) || overlaps(x.data, x.bytes, output, bytes) || overlaps(w.data, raw, output, bytes))
        throw std::invalid_argument("Gemma T1 striped Q4 linear output lifetime");
    launch<false>(w, w, x, output, nullptr, nullptr, stream);
}

void pair_single_q4_striped_t1(primitive::Matrix a, primitive::Matrix b, SingleQ8View x,
    float* gate, float* up, float* activation, cudaStream_t stream) {
    validate(a, x, stream); validate(b, x, stream);
    if (a.rows != b.rows || a.columns != b.columns)
        throw std::invalid_argument("Gemma T1 striped Q4 pair shape");
    const size_t bytes = size_t(a.rows) * sizeof(float), raw = size_t(a.rows) * (a.columns / 256) * 144;
    const void* outputs[]{gate, up, activation};
    for (int i = 0; i < 3; ++i) {
        if (!aligned(outputs[i]) || overlaps(x.data, x.bytes, outputs[i], bytes) ||
            overlaps(a.data, raw, outputs[i], bytes) || overlaps(b.data, raw, outputs[i], bytes))
            throw std::invalid_argument("Gemma T1 striped Q4 pair input/weight alias");
        for (int j = i + 1; j < 3; ++j)
            if (overlaps(outputs[i], bytes, outputs[j], bytes))
                throw std::invalid_argument("Gemma T1 striped Q4 pair output alias");
    }
    launch<true>(a, b, x, gate, up, activation, stream);
}
}
