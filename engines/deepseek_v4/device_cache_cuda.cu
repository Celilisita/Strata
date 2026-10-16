#include "device_cache_cuda.hpp"
#include "device_cache_protocol.hpp"
#include "ggml-cuda/common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include <climits>
#include <cstring>

namespace {
using strata::deepseek_v4::DeviceMissPacket;
using strata::deepseek_v4::device_frequency_scores;
__device__ unsigned long long nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
__device__ bool probe_routes(const volatile int32_t* directory, const int32_t* ids, int32_t* slots,
                             int32_t* status, unsigned long long* counters) {
    atomicAdd(counters, 1ull);
    bool missing = false;
    int hits = 0;
    for (int i = 0; i < 6; ++i) {
        const int id = ids[i];
        if (id < 0 || id >= 256) { atomicExch(status, 1); slots[i] = 0; missing = true; }
        else { slots[i] = directory[id]; missing |= slots[i] < 0; hits += slots[i] >= 0; }
    }
    atomicAdd(counters + 2, static_cast<unsigned long long>(hits));
    const bool cold = missing || *status;
    if (!cold) atomicAdd(counters + 1, 1ull);
    return cold;
}
__device__ void observe_frequency(uint32_t* frequency, int layer, const int32_t* ids, size_t score_count) {
    for (int i = 0; i < 6; ++i) {
        if (++frequency[score_count] == 65536) {
            for (size_t key = 0; key < score_count; ++key) frequency[key] /= 2;
            frequency[score_count] = 0;
        }
        auto& score = frequency[layer * 256 + ids[i]];
        if (score < 65535) ++score;
    }
}
__global__ void lookup_kernel(const volatile int32_t* directory, const int32_t* ids, int32_t* slots,
                              int32_t* status, unsigned long long* counters,
                              DeviceMissPacket* packet, int layer, unsigned long long timeout, uint32_t* frequency, size_t score_count) {
    if (!frequency) {
        if (threadIdx.x != 0 || !probe_routes(directory, ids, slots, status, counters)) return;
    } else {
        __shared__ bool cold;
        if (threadIdx.x == 0) {
            cold = probe_routes(directory, ids, slots, status, counters);
            if (!*status) observe_frequency(frequency, layer, ids, score_count);
        }
        __syncthreads();
        if (!cold) return; // Allhit keeps history entirely in device memory.
        if (!*status) {
            // Publish from this same lookup. Enqueueing a separate D2H copy
            // while the compute stream awaits this miss can block in WDDM.
            for (size_t key = threadIdx.x; key <= score_count; key += blockDim.x)
                packet->frequency[key] = frequency[key];
            // Each writer must flush its own mapped writes before thread0
            // publishes the ticket; a fence in thread0 alone is insufficient.
            __threadfence_system();
            __syncthreads();
        }
        if (threadIdx.x != 0) return;
    }
    if (!*status) {
        packet->native_request = 0;
        packet->layer = unsigned(layer);
        for (int i = 0; i < 6; ++i) packet->ids[i] = ids[i];
        const unsigned sequence = packet->sequence + 1;
        __threadfence_system();
        packet->sequence = sequence;
        __threadfence_system();
        atomicAdd(counters + 3, 1ull);
        const unsigned long long begin = nanoseconds();
        while (packet->response != sequence) {
            if (nanoseconds() - begin >= timeout) {
                atomicExch(status, 1);
                atomicAdd(counters + 4, 1ull);
                break;
            }
            __nanosleep(1000);
        }
        __threadfence_system();
        if (packet->error) atomicExch(status, 1);
    }
    for (int i = 0; i < 6; ++i) {
        const int id = ids[i];
        const int slot = id >= 0 && id < 256 ? directory[id] : -1;
        if (*status || slot < 0) { slots[i] = 0; atomicExch(status, 1); }
        else slots[i] = slot;
    }
}
__global__ void guard_kernel(const float* input, float* output, const int32_t* status, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = *status ? __int_as_float(0x7fc00000) : input[i];
}
}
bool strata_device_cache_lookup_cuda(ggml_backend_cuda_context& context, ggml_tensor* tensor) {
    using namespace strata::deepseek_v4;
    if (!device_cache_lookup(tensor)) return false;
    GGML_ASSERT(tensor->type == GGML_TYPE_I32 && ggml_nelements(tensor) == 6 && ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->src[0]->type == GGML_TYPE_I32 && ggml_nelements(tensor->src[0]) == 256 &&
                tensor->src[1]->type == GGML_TYPE_I32 && ggml_nelements(tensor->src[1]) == 6 &&
                tensor->src[2]->type == GGML_TYPE_I32 && ggml_nelements(tensor->src[2]) == 1 &&
                tensor->src[3]->type == GGML_TYPE_I64 && ggml_nelements(tensor->src[3]) == device_cache_counter_count);
    for (int i = 0; i < 4; ++i) GGML_ASSERT(ggml_is_contiguous(tensor->src[i]));
    uintptr_t packet = 0;
    uint64_t timeout = 0;
    std::memcpy(&packet, tensor->op_params, sizeof(packet));
    std::memcpy(&timeout, tensor->op_params + 4, sizeof(timeout));
    GGML_ASSERT(packet && timeout && tensor->op_params[2] >= 0 && tensor->op_params[2] < device_cache_max_layers);
    lookup_kernel<<<1, 32, 0, context.stream()>>>(static_cast<const int32_t*>(tensor->src[0]->data),
        static_cast<const int32_t*>(tensor->src[1]->data), static_cast<int32_t*>(tensor->data),
        static_cast<int32_t*>(tensor->src[2]->data), static_cast<unsigned long long*>(tensor->src[3]->data),
        reinterpret_cast<DeviceMissPacket*>(packet), tensor->op_params[2], timeout,
        tensor->src[4] ? static_cast<uint32_t*>(tensor->src[4]->data) : nullptr, tensor->src[4] ? size_t(tensor->src[4]->ne[0]-1) : 0);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
bool strata_device_cache_guard_cuda(ggml_backend_cuda_context& context, ggml_tensor* tensor) {
    using namespace strata::deepseek_v4;
    if (!device_cache_guard(tensor)) return false;
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && tensor->src[0]->type == GGML_TYPE_F32 &&
                tensor->src[1]->type == GGML_TYPE_I32 && ggml_nelements(tensor->src[1]) == 1 &&
                ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0]) && ggml_is_contiguous(tensor->src[1]) &&
                ggml_nelements(tensor) == ggml_nelements(tensor->src[0]) && ggml_nelements(tensor) <= INT_MAX);
    const int count = int(ggml_nelements(tensor));
    guard_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(static_cast<const float*>(tensor->src[0]->data),
        static_cast<float*>(tensor->data), static_cast<const int32_t*>(tensor->src[1]->data), count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// Prepare the canonical pool before a device lookup can wait for host DMA.
// Expanding VMM after that lookup may synchronize the device and block its miss worker.
bool strata_device_cache_reserve_cuda(ggml_backend_t backend, size_t bytes, bool all_streams) {
    if (!backend || !ggml_backend_is_cuda(backend) || bytes < (1ull << 20) || bytes > (1ull << 30)) return false;
    auto* context = static_cast<ggml_backend_cuda_context*>(backend->context);
    if (!ggml_cuda_info().devices[context->device].vmm) return false;
    ggml_cuda_set_device(context->device);
    using namespace strata::deepseek_v4;
    const int previous_stream = context->curr_stream_no;
    const int count = all_streams ? GGML_CUDA_MAX_STREAMS : 1;
    cache_graph_phase(context, CacheGraphPhase::Prepare, 0);
    for (int stream = 0; stream < count; ++stream) {
        context->curr_stream_no = stream;
        cache_graph_stream(uint32_t(stream));
        context->stream();
        if (all_streams) context->cublas_handle();
        auto& pool = context->pool();
        size_t actual = 0;
        const size_t reserve = stream == 0 ? bytes : std::min(bytes, size_t(32ull << 20));
        void* storage = pool.alloc(reserve, &actual);
        pool.free(storage, actual);
    }
    context->curr_stream_no = previous_stream;
    cache_graph_phase(context, CacheGraphPhase::Idle, uint32_t(previous_stream));
    return true;
}

bool strata_device_cache_trace_cuda(ggml_backend_t backend, strata::deepseek_v4::CacheGraphTrace* trace) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    return strata::deepseek_v4::bind_cache_graph_trace(backend->context, trace);
}
