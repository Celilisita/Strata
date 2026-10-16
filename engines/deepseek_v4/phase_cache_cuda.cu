#include "phase_cache_cuda.hpp"
#include "ggml-cuda/common.cuh"
#include "ggml-backend-impl.h"
namespace strata::deepseek_v4 {
namespace {
__global__ void swap(uint32_t* gate,uint32_t* up,uint32_t* down,size_t words,int a,int b) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words*3;i+=size_t(gridDim.x)*blockDim.x) {
        auto* bank=i<words?gate:i<words*2?up:down;const size_t offset=i%words;
        const auto old=bank[size_t(a)*words+offset];
        bank[size_t(a)*words+offset]=bank[size_t(b)*words+offset];bank[size_t(b)*words+offset]=old;
    }
}
}
void phase_swap_rows(ggml_backend_t backend,void* gate,void* up,void* down,size_t bytes,int a,int b) {
    GGML_ASSERT(backend&&gate&&up&&down&&bytes%4==0&&a>=0&&b>=0);
    auto* ctx=static_cast<ggml_backend_cuda_context*>(backend->context);ggml_cuda_set_device(ctx->device);
    swap<<<512,128,0,ctx->stream()>>>(static_cast<uint32_t*>(gate),static_cast<uint32_t*>(up),static_cast<uint32_t*>(down),bytes/4,a,b);
    CUDA_CHECK(cudaGetLastError());
}
}
