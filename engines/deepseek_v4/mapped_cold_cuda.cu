#include "mapped_cold_packet.hpp"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include <cuda/atomic>
#include <cstring>
#include <new>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
using SystemWord = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;

__global__ void publish_packet(const float* norm, const int32_t* ids, MappedColdPacket* packet,
                                uint32_t* state, uint32_t mask, uint32_t layer) {
    __shared__ uint32_t ticket;
    if (threadIdx.x == 0) {
        state[1] = mask;
        ticket = 0;
        if (!state[0] && mask) {
            for (int rank = 0; rank < 6; ++rank)
                if (ids[rank] < 0 || ids[rank] >= 256) state[0] = 1;
            if (layer >= 43 || state[2] == UINT32_MAX) state[0] = 2;
            if (!state[0]) ticket = ++state[2];
        }
    }
    __syncthreads();
    if (!ticket) return; // All-hit never accesses any mapped packet field.
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) packet->activation[i] = norm[i];
    if (threadIdx.x < 6) packet->ids[threadIdx.x] = ids[threadIdx.x];
    if (threadIdx.x == 0) { packet->layer = layer; packet->mask = mask; }
    // Every producer thread flushes its own mapped stores before publication.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) SystemWord(packet->ready).store(ticket, cuda::memory_order_release);
}

__global__ void join_packet(MappedColdPacket* packet, uint32_t* state, float* down,
                            uint64_t timeout_ticks) {
    const uint32_t mask = state[1];
    if (!mask && !state[0]) return;
    if (threadIdx.x == 0 && !state[0]) {
        const uint64_t start = clock64();
        const uint32_t ticket = state[2];
        while (SystemWord(packet->done).load(cuda::memory_order_acquire) != ticket) {
            if (clock64() - start > timeout_ticks) { state[0] = 3; break; }
            __nanosleep(64);
        }
        if (!state[0] && packet->error) state[0] = 4;
    }
    __syncthreads();
    if (state[0]) {
        for (int i = threadIdx.x; i < 6 * 4096; i += blockDim.x) down[i] = __int_as_float(0x7fc00000);
        return;
    }
    // Each consumer uses an acquire load before reading its mapped contribution.
    (void) SystemWord(packet->done).load(cuda::memory_order_acquire);
    for (int i = threadIdx.x; i < 6 * 4096; i += blockDim.x) {
        const int rank = i / 4096;
        if (mask & (1u << rank)) {
            const uint32_t previous = mask & ((1u << rank) - 1);
            const int compact = __popc(previous);
            down[i] = packet->contribution[compact * 4096 + i % 4096];
        }
    }
}
}

void mapped_cold_create(ggml_backend_t backend, uint32_t timeout_ms, MappedColdStorage& out) {
    if (!ggml_backend_is_cuda(backend) || timeout_ms < 1 || timeout_ms > 500 || out.host)
        throw std::runtime_error("mapped cold carrier backend/timeout/storage contract differs");
    auto* context = static_cast<ggml_backend_cuda_context*>(backend->context);
    out.device_id = context->device;
    ggml_cuda_set_device(out.device_id);
    int mapped = 0, unified = 0, clock = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mapped, cudaDevAttrCanMapHostMemory, out.device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, out.device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, out.device_id));
    if (!mapped || !unified || clock <= 0) throw std::runtime_error("mapped cold carrier requires mapped/UVA support");
    out.stream = context->stream();
    out.timeout_ticks = uint64_t(clock) * timeout_ms;
    try {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&out.host), sizeof(MappedColdPacket),
                                cudaHostAllocMapped | cudaHostAllocPortable));
        new (out.host) MappedColdPacket{};
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&out.device), out.host, 0));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&out.state), 3 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(out.state, 0, 3 * sizeof(uint32_t), context->stream()));
        CUDA_CHECK(cudaStreamSynchronize(context->stream()));
    } catch (...) { mapped_cold_destroy(out); throw; }
}
void mapped_cold_synchronize(const MappedColdStorage& storage) {
    ggml_cuda_set_device(storage.device_id);
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(storage.stream)));
}
void mapped_cold_destroy(MappedColdStorage& storage) noexcept {
    if (storage.device_id >= 0) {
        ggml_cuda_set_device(storage.device_id);
        if (storage.stream) (void) cudaStreamSynchronize(static_cast<cudaStream_t>(storage.stream));
        if (storage.state) (void) cudaFree(storage.state);
        if (storage.host) (void) cudaFreeHost(storage.host);
    }
    storage = {};
}
uint32_t mapped_cold_ready(const MappedColdPacket* packet) {
    return SystemWord(const_cast<uint32_t&>(packet->ready)).load(cuda::memory_order_acquire);
}
void mapped_cold_done(MappedColdPacket* packet, uint32_t ticket) {
    SystemWord(packet->done).store(ticket, cuda::memory_order_release);
}
void mapped_cold_enqueue(const MappedColdStorage& storage, const ggml_tensor* norm,
                         const ggml_tensor* ids, ggml_tensor* down, uint32_t mask, uint32_t layer) {
    if (!storage.device || !storage.state || mask > 63 || layer >= 43 || !norm || !ids || !down ||
        norm->type != GGML_TYPE_F32 || ggml_nelements(norm) != 4096 || !ggml_is_contiguous(norm) ||
        ids->type != GGML_TYPE_I32 || ggml_nelements(ids) != 6 || !ggml_is_contiguous(ids) ||
        down->type != GGML_TYPE_F32 || ggml_nelements(down) != 6 * 4096 || !ggml_is_contiguous(down))
        throw std::runtime_error("mapped cold carrier actual layer shape differs");
    ggml_cuda_set_device(storage.device_id);
    const auto stream = static_cast<cudaStream_t>(storage.stream);
    publish_packet<<<1, 128, 0, stream>>>(static_cast<const float*>(norm->data),
        static_cast<const int32_t*>(ids->data), storage.device, storage.state, mask, layer);
    join_packet<<<1, 128, 0, stream>>>(storage.device, storage.state, static_cast<float*>(down->data), storage.timeout_ticks);
    CUDA_CHECK(cudaGetLastError());
}
uint32_t mapped_cold_status(const MappedColdStorage& storage) {
    uint32_t result = UINT32_MAX;
    ggml_cuda_set_device(storage.device_id);
    const auto stream = static_cast<cudaStream_t>(storage.stream);
    CUDA_CHECK(cudaMemcpyAsync(&result, storage.state, sizeof(result), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}
}
