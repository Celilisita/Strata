#include "strata/qwen35/native/ops.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "math.cuh"
#include <cmath>
namespace strata::qwen35::native {
namespace {
__device__ float sum_warp(float value) {
    for (int d = 16; d; d /= 2) value = __fadd_rn(value, __shfl_xor_sync(0xffffffff, value, d));
    return value;
}
__global__ void rms(const float *x, const float *w, float *out, const float *gate, int features, float epsilon, int stride) {
    __shared__ float partial[32];
    const size_t base = size_t(blockIdx.x) * features;
    const size_t source = size_t(blockIdx.x) * stride;
    float square = 0;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        const float value = x[source + i];
        square = __fmaf_rn(value, value, square);
    }
    square = sum_warp(square);
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = square;
    __syncthreads();
    square = sum_warp(threadIdx.x % 32 < blockDim.x / 32 ? partial[threadIdx.x % 32] : 0.0f);
    const float scale = rsqrtf(__fmaf_rn(square, __fdividef(1.0f, float(features)), epsilon));
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float value = __fmul_rn(__fmul_rn(x[source + i], scale), w[i]);
        if (gate) value = __fmul_rn(value, swiglu_value(gate[base + i], 1.0f));
        out[base + i] = value;
    }
}
__global__ void residual(const float *a, const float *b, float *out, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __fadd_rn(a[i], b[i]);
}
}
void rms_norm(const float *x, const float *w, float *out, int features, int rows, float eps,
              cudaStream_t stream, const float *gate, int input_stride) {
    if (!input_stride) input_stride = features;
    if (!x || !w || !out || features < 32 || features > 32768 || rows < 1 || rows > 2097152 || !(eps > 0) || !std::isfinite(eps))
        throw std::invalid_argument("invalid own RMS normalization");
    if (input_stride < features || (gate && input_stride != features)) throw std::invalid_argument("invalid own RMS row stride");
    rms<<<rows, features >= 1024 ? 1024 : 256, 0, stream>>>(x, w, out, gate, features, eps, input_stride);
    check_cuda(cudaGetLastError(), "native RMS normalization");
}
void add(const float *a, const float *b, float *out, size_t count, cudaStream_t stream) {
    if (!a || !b || !out || !count) throw std::invalid_argument("invalid own residual addition");
    residual<<<unsigned((count + 255) / 256), 256, 0, stream>>>(a, b, out, count);
    check_cuda(cudaGetLastError(), "native residual addition");
}
}
