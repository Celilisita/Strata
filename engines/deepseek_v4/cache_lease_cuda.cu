#include "cache_lease_cuda.hpp"
#include "cache_lease_protocol.hpp"
#include "ggml-cuda/common.cuh"
#include <climits>

namespace {
__global__ void lease_lookup_kernel(const int32_t* safe, const int32_t* valid,
                                    const int32_t* ids, int32_t* slots, int32_t* missing, int count) {
    const int index = threadIdx.x;
    if (index >= count) return;
    const int32_t id = ids[index];
    const bool in_range = id >= 0 && id < 256;
    const bool hit = in_range && valid[id] != 0;
    slots[index] = in_range ? safe[id] : 0;
    if (!hit) atomicOr(reinterpret_cast<unsigned int*>(missing), 1u);
}

__global__ void lease_guard_kernel(const float* input, float* output, const int32_t* missing, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = *missing ? __int_as_float(0x7fc00000) : input[index];
}
}

bool strata_cache_lease_lookup_cuda(ggml_backend_cuda_context& context, ggml_tensor* tensor) {
    using namespace strata::deepseek_v4;
    if (!lease_lookup(tensor)) return false;
    const auto* safe = tensor->src[0];
    const auto* ids = tensor->src[1];
    const auto* valid = tensor->src[2];
    auto* status = tensor->src[3];
    GGML_ASSERT(safe && ids && valid && status && tensor->type == GGML_TYPE_I32);
    GGML_ASSERT(safe->type == GGML_TYPE_I32 && ids->type == GGML_TYPE_I32 &&
                valid->type == GGML_TYPE_I32 && status->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(safe) && ggml_is_contiguous(ids) && ggml_is_contiguous(valid) &&
                ggml_is_contiguous(status) && ggml_is_contiguous(tensor));
    const int count = int(ggml_nelements(ids));
    GGML_ASSERT(ggml_nelements(safe) == 256 && ggml_nelements(valid) == 256 &&
                ggml_nelements(status) == 1 && count > 0 && count <= 48 && ggml_nelements(tensor) == count);
    lease_lookup_kernel<<<1, 64, 0, context.stream()>>>(static_cast<const int32_t*>(safe->data),
        static_cast<const int32_t*>(valid->data), static_cast<const int32_t*>(ids->data),
        static_cast<int32_t*>(tensor->data), static_cast<int32_t*>(status->data), count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool strata_cache_lease_guard_cuda(ggml_backend_cuda_context& context, ggml_tensor* tensor) {
    using namespace strata::deepseek_v4;
    if (!lease_guard(tensor)) return false;
    const auto* input = tensor->src[0];
    const auto* status = tensor->src[1];
    GGML_ASSERT(input && status && input->type == GGML_TYPE_F32 && tensor->type == GGML_TYPE_F32 &&
                status->type == GGML_TYPE_I32 && ggml_nelements(status) == 1 &&
                ggml_is_contiguous(input) && ggml_is_contiguous(tensor) && ggml_is_contiguous(status));
    GGML_ASSERT(ggml_nelements(input) == ggml_nelements(tensor) && ggml_nelements(tensor) <= INT_MAX);
    const int count = int(ggml_nelements(tensor));
    lease_guard_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(
        static_cast<const float*>(input->data), static_cast<float*>(tensor->data),
        static_cast<const int32_t*>(status->data), count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
