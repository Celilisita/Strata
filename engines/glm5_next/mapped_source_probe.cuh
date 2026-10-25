#pragma once
// Included after the standard CUDA buffer interface in ggml-cuda.cu. No
// production registration, dispatch, stream, lease or kernel changes.
#include "mapped_source_probe.hpp"
#include <limits>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace strata::glm5_next {
namespace mapped_source_detail {
struct Context : ggml_backend_cuda_buffer_context {
    void* host;
    Context(int device, void* alias, void* source)
        : ggml_backend_cuda_buffer_context(device, alias), host(source) {}
};
inline Context* context(ggml_backend_buffer_t buffer) {
    auto* base = static_cast<ggml_backend_cuda_buffer_context*>(buffer->context);
    return static_cast<Context*>(base);
}
inline void free_buffer(ggml_backend_buffer_t buffer) {
    auto* ctx = context(buffer);
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaHostUnregister(ctx->host));
    // The standard context destructor calls cudaFree. Never pass a host alias.
    ctx->dev_ptr = nullptr;
    delete ctx;
}
inline bool contains(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, size_t bytes) {
    const auto base = reinterpret_cast<uintptr_t>(context(buffer)->dev_ptr);
    const auto address = reinterpret_cast<uintptr_t>(tensor->data);
    return address >= base && address - base <= buffer->size && bytes <= buffer->size - (address - base);
}
inline ggml_status init_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    // Standard init_tensor can zero quant padding. This source is immutable;
    // only layouts needing no extra padding are admitted, with no page writes.
    if (!tensor || ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        ggml_backend_buffer_get_alloc_size(buffer, tensor) != ggml_nbytes(tensor) ||
        !contains(buffer, tensor, ggml_nbytes(tensor))) return GGML_STATUS_FAILED;
    return GGML_STATUS_SUCCESS;
}
inline void get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* tensor,
                       void* output, size_t offset, size_t bytes) {
    GGML_ASSERT(offset <= ggml_nbytes(tensor) && bytes <= ggml_nbytes(tensor) - offset);
    GGML_ASSERT(contains(buffer, tensor, offset + bytes));
    ggml_cuda_set_device(context(buffer)->device);
    CUDA_CHECK(cudaMemcpyAsync(output, static_cast<const char*>(tensor->data) + offset,
                               bytes, cudaMemcpyDefault, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
inline void get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* output,
                          size_t offset, size_t bytes, size_t copies, size_t tensor_stride, size_t output_stride) {
    if (!copies || !bytes) return;
    GGML_ASSERT(offset <= ggml_nbytes(tensor) && bytes <= ggml_nbytes(tensor) - offset);
    GGML_ASSERT(copies - 1 <= (ggml_nbytes(tensor) - offset - bytes) / (tensor_stride ? tensor_stride : 1));
    const size_t span = offset + (copies - 1) * tensor_stride + bytes;
    GGML_ASSERT(contains(buffer, tensor, span));
    ggml_cuda_set_device(context(buffer)->device);
    CUDA_CHECK(cudaMemcpy2DAsync(output, output_stride, static_cast<const char*>(tensor->data) + offset,
                                 tensor_stride, bytes, copies, cudaMemcpyDefault, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
}
}
#endif

static ggml_backend_buffer_t strata_glm53_probe_map_original_host(ggml_backend_t backend,
    void* host, size_t bytes, strata::glm5_next::MappedSourceInfo* info) {
    if (!info) return nullptr;
    *info = {};
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    (void) backend; (void) host; (void) bytes;
    info->cuda_error = 801; // The fixture bridge is CUDA-only.
    return nullptr;
#else
    using namespace strata::glm5_next;
    info->flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
    bool registered = false;
    const auto fail = [&](cudaError_t error) -> ggml_backend_buffer_t {
        info->cuda_error = uint32_t(error);
        if (registered) {
            // No consumer can have seen this failed mapping. Cleanup failure
            // is fatal rather than returning an unowned live registration.
            CUDA_CHECK(cudaHostUnregister(host));
        }
        return nullptr; // Do not clear any CUDA last-error state.
    };
    if (!backend || !ggml_backend_is_cuda(backend) || !host || !bytes ||
        bytes > (std::numeric_limits<uintptr_t>::max)() - reinterpret_cast<uintptr_t>(host))
        return fail(cudaErrorInvalidValue);
    auto* cuda = static_cast<ggml_backend_cuda_context*>(backend->context);
    const int physical = ggml_cuda_get_physical_device(cuda->device);
    auto status = cudaSetDevice(physical);
    if (status != cudaSuccess) return fail(status);
    status = cudaDeviceGetAttribute(&info->can_map, cudaDevAttrCanMapHostMemory, physical);
    if (status != cudaSuccess) return fail(status);
    status = cudaDeviceGetAttribute(&info->unified, cudaDevAttrUnifiedAddressing, physical);
    if (status != cudaSuccess) return fail(status);
    status = cudaDeviceGetAttribute(&info->host_pointer_supported, cudaDevAttrCanUseHostPointerForRegisteredMem, physical);
    if (status != cudaSuccess) return fail(status);
    status = cudaDeviceGetAttribute(&info->l2_bytes, cudaDevAttrL2CacheSize, physical);
    if (status != cudaSuccess) return fail(status);
    unsigned flags = 0;
    status = cudaGetDeviceFlags(&flags);
    if (status != cudaSuccess) return fail(status);
    info->device_flags = int(flags);
    if (!info->can_map || !info->unified) return fail(cudaErrorNotSupported);
    status = cudaHostRegister(host, bytes, info->flags);
    if (status != cudaSuccess) return fail(status);
    registered = true;
    void* alias = nullptr;
    status = cudaHostGetDevicePointer(&alias, host, 0);
    if (status != cudaSuccess) return fail(status);
    cudaPointerAttributes original{}, mapped{};
    status = cudaPointerGetAttributes(&original, host);
    if (status != cudaSuccess) return fail(status);
    status = cudaPointerGetAttributes(&mapped, alias);
    if (status != cudaSuccess) return fail(status);
    if (!alias || original.type != cudaMemoryTypeHost || mapped.type != cudaMemoryTypeHost ||
        original.hostPointer != host || mapped.hostPointer != host ||
        original.devicePointer != alias || mapped.devicePointer != alias ||
        original.device != physical || mapped.device != physical)
        return fail(cudaErrorInvalidValue);
    auto iface = ggml_backend_cuda_buffer_interface;
    iface.free_buffer = mapped_source_detail::free_buffer;
    iface.init_tensor = mapped_source_detail::init_tensor;
    iface.memset_tensor = nullptr; iface.set_tensor = nullptr; iface.set_tensor_2d = nullptr;
    iface.get_tensor = mapped_source_detail::get_tensor;
    iface.get_tensor_2d = mapped_source_detail::get_tensor_2d;
    iface.cpy_tensor = nullptr; iface.clear = nullptr; iface.reset = nullptr;
    mapped_source_detail::Context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    try {
        ctx = new mapped_source_detail::Context(cuda->device, alias, host);
        buffer = ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(cuda->device), iface,
            static_cast<ggml_backend_cuda_buffer_context*>(ctx), bytes);
    } catch (...) {
        if (ctx) { ctx->dev_ptr = nullptr; delete ctx; }
        return fail(cudaErrorMemoryAllocation);
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    info->device_base = alias; info->registered_bytes = bytes;
    return buffer;
#endif
}
