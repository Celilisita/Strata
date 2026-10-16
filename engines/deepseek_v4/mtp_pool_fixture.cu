#include "ggml-cuda/common.cuh"
#include "ggml-backend-impl.h"
// Test-only allocation, linked into the bounded component executable.
// Its RAII loan is returned before retirement; no model payload is replaced.
void mtp_pool_fixture(ggml_backend_t backend,size_t bytes) {
    auto* context=static_cast<ggml_backend_cuda_context*>(backend->context);
    ggml_cuda_set_device(context->device);
    ggml_cuda_pool_alloc<uint8_t> scratch(context->pool(),bytes);
    CUDA_CHECK(cudaMemsetAsync(scratch.get(),0,4096,context->stream()));
    CUDA_CHECK(cudaMemsetAsync(scratch.get()+bytes-4096,0,4096,context->stream()));
    CUDA_CHECK(cudaStreamSynchronize(context->stream()));
}
