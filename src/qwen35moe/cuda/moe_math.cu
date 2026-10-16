#include "strata/qwen35moe/cuda_kernels.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
template<bool Masked>
__global__ void swiglu_kernel(const float * __restrict__ gate,
                              const float * __restrict__ up,
                              float * __restrict__ hidden, int width,
                              const unsigned long long* addresses, int row_width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width && (!Masked || addresses[i / row_width])) {
        const float g = gate[i];
        hidden[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void gate_shared_kernel(float * values, const float * logit, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        const float weight = 1.0f / (1.0f + expf(-logit[0]));
        values[i] *= weight;
    }
}
}

void swiglu(const float * gate, const float * up, float * hidden, int width, void * stream,
              const unsigned long long* addresses, int row_width) {
    if (!gate || !up || !hidden || !stream || width <= 0 ||
        (addresses && (row_width <= 0 || width % row_width)))
        throw std::invalid_argument("invalid qwen35moe SwiGLU arguments");
    if (addresses)
        swiglu_kernel<true><<<unsigned((width + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            gate, up, hidden, width, addresses, row_width);
    else
        swiglu_kernel<false><<<unsigned((width + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            gate, up, hidden, width, nullptr, 0);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void gate_shared(float * values, const float * logit, int width, void * stream) {
    if (!values || !logit || !stream || width <= 0)
        throw std::invalid_argument("invalid Qwen3.5 shared expert gate");
    gate_shared_kernel<<<unsigned((width + 255) / 256), 256, 0,
                         static_cast<cudaStream_t>(stream)>>>(values, logit, width);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace strata::qwen35moe
