#include "strata/qwen35moe/router_tiled.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <array>
#include <utility>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <cstring>
namespace strata::qwen35moe {
namespace {
__global__ void project_four(const float* __restrict__ rows, const float* __restrict__ x,
                            float* __restrict__ logits, int width, int outputs, int tokens) {
    const int row = int(blockIdx.x), first = 4 * int(blockIdx.y), lane = int(threadIdx.x);
    float sums[4]{};
    for (int i = lane; i < width; i += 128) {
        const float weight = rows[size_t(row) * width + i];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (first + j < tokens) sums[j] = fmaf(weight, x[size_t(first + j) * width + i], sums[j]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mask = 16; mask; mask >>= 1)
            sums[j] += __shfl_xor_sync(0xffffffffu, sums[j], mask);
    __shared__ float partial[4][4];
    if ((lane & 31) == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) partial[j][lane / 32] = sums[j];
    }
    __syncthreads();
    if (lane < 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float sum = lane < 4 ? partial[j][lane] : 0.0f;
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask);
            if (lane == 0 && first + j < tokens) logits[size_t(first + j) * outputs + row] = sum;
        }
    }
}
#include "router_tf32x3.cuh"
}
void route_project_tiled_f32(const float* rows, const float* input, float* logits,
                        int width, int outputs, int tokens, void* stream) {
    if (!rows || !input || !logits || !stream || width <= 0 || width % 128 ||
        outputs <= 0 || outputs > 65535 || tokens < 4 || tokens > 8192)
        throw std::invalid_argument("invalid canonical tiled F32 router");
    validate_router_spans(rows,input,logits,width,outputs,tokens);
    project_four<<<dim3(unsigned(outputs), unsigned((tokens + 3) / 4)), 128, 0,
        static_cast<cudaStream_t>(stream)>>>(rows, input, logits, width, outputs, tokens);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void route_project_tf32x3(const float* rows,const float* input,float* logits,
                         int width,int outputs,int tokens,void* stream) {
    if(!stream || width!=2048 || outputs!=257 || tokens<4 || tokens>4096)
        throw std::invalid_argument("router TF32x3 requires K2048/N257/tokens4..4096");
    validate_router_spans(rows,input,logits,width,outputs,tokens);
    prepare_router_tf32x3();
    project_tf32x3<<<dim3(9,unsigned((tokens+31)/32)),256,0,static_cast<cudaStream_t>(stream)>>>(
        rows,input,logits,tokens);
    const auto e=cudaGetLastError();
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void route_project_tiled(const float* rows,const float* input,float* logits,
                         int width,int outputs,int tokens,void* stream) {
    static const bool enabled=router_tf32x3_policy(std::getenv("STRATA_QWEN35_ROUTER_TF32X3"));
    // Prefix256 and whole-route must use the same per-row math and tile origin.
    if(enabled && width==2048 && outputs==257 && tokens>=256 && tokens<=4096)
        route_project_tf32x3(rows,input,logits,width,outputs,tokens,stream);
    else route_project_tiled_f32(rows,input,logits,width,outputs,tokens,stream);
}
RouterProjectResources route_project_resources(bool tensor) {
    if(tensor)prepare_router_tf32x3();
    cudaFuncAttributes a{};int active=0;cudaError_t e;
    if(tensor) {
        e=cudaFuncGetAttributes(&a,project_tf32x3);
        if(e==cudaSuccess)e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,project_tf32x3,256,0);
    } else {
        e=cudaFuncGetAttributes(&a,project_four);
        if(e==cudaSuccess)e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,project_four,128,0);
    }
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    return {a.numRegs,tensor?256:128,active,a.sharedSizeBytes,a.localSizeBytes};
}
} // namespace strata::qwen35moe
