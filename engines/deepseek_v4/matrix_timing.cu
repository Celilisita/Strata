#include "matrix_timing.hpp"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
struct Events {
    cudaEvent_t start = nullptr, end = nullptr;
    ~Events() {
        if (start) cudaEventDestroy(start);
        if (end) cudaEventDestroy(end);
    }
};
void compute(ggml_backend_t backend, ggml_cgraph* graph) {
    if (ggml_backend_graph_compute_async(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("isolated canonical matrix graph failed");
}
__global__ void read_eviction(const volatile uint32_t* data, uint32_t* checksum, size_t words) {
    uint32_t value = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x)
        value ^= data[i];
    for (int offset = 16; offset; offset >>= 1)
        value ^= __shfl_xor_sync(0xffffffffu, value, offset);
    if ((threadIdx.x & 31) == 0) atomicXor(checksum, value);
}
struct Eviction {
    uint32_t* data = nullptr;
    uint32_t* checksum = nullptr;
    size_t bytes = 0;
    ~Eviction() {
        if (checksum) cudaFree(checksum);
        if (data) cudaFree(data);
    }
    void prepare(int device, cudaStream_t stream) {
        int l2 = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
        if (l2 <= 0 || size_t(l2) > (128ull << 20))
            throw std::runtime_error("L2 eviction exceeds bounded probe budget");
        bytes = size_t(l2) * 4;
        CUDA_CHECK(cudaMalloc(&data, bytes));
        CUDA_CHECK(cudaMalloc(&checksum, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(data, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(checksum, 0, sizeof(uint32_t), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    void enqueue(cudaStream_t stream) { read_eviction<<<1024, 256, 0, stream>>>(data, checksum, bytes / 4); }
};
}
std::array<double, 3> measure_matrix_graph(ggml_backend_t backend, ggml_cgraph* graph, bool l2_cold) {
    auto* context = static_cast<ggml_backend_cuda_context*>(backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    const auto stream = context->stream();
    for (int i = 0; i < 3; ++i) { compute(backend, graph); ggml_backend_synchronize(backend); }
    Events events;
    CUDA_CHECK(cudaEventCreate(&events.start));
    CUDA_CHECK(cudaEventCreate(&events.end));
    std::array<double, 3> result{};
    if (l2_cold) {
        Eviction eviction;
        eviction.prepare(context->device, stream);
        std::array<Events, 16> measurements;
        for (auto& pair : measurements) {
            CUDA_CHECK(cudaEventCreate(&pair.start));
            CUDA_CHECK(cudaEventCreate(&pair.end));
        }
        for (auto& sample : result) {
            for (auto& pair : measurements) {
                eviction.enqueue(stream);
                CUDA_CHECK(cudaEventRecord(pair.start, stream));
                compute(backend, graph);
                CUDA_CHECK(cudaEventRecord(pair.end, stream));
            }
            CUDA_CHECK(cudaEventSynchronize(measurements.back().end));
            double total = 0;
            for (auto& pair : measurements) {
                float milliseconds = 0;
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, pair.start, pair.end));
                total += milliseconds;
            }
            sample = total / measurements.size();
        }
        return result;
    }
    for (auto& sample : result) {
        CUDA_CHECK(cudaEventRecord(events.start, stream));
        for (int replay = 0; replay < 32; ++replay) compute(backend, graph);
        CUDA_CHECK(cudaEventRecord(events.end, stream));
        CUDA_CHECK(cudaEventSynchronize(events.end));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, events.start, events.end));
        sample = double(milliseconds) / 32;
    }
    return result;
}
}
