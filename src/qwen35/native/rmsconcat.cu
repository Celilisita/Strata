#include "rmsconcat.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <math.cuh>
#include <cmath>
#include <cstdint>
#include <limits>
namespace strata::qwen35::native {
namespace {
// Original RMS XOR/RN body; only the final store address has a row stride and half offset.
__device__ float sum_warp(float value) {
    for (int d = 16; d; d /= 2) value = __fadd_rn(value, __shfl_xor_sync(0xffffffff, value, d));
    return value;
}
__global__ void rmsconcat_kernel(const float *x, const float *w, float *out, const float *gate, int features, float epsilon, int stride, int offset) {
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
        out[2 * base + offset + i] = value;
    }
}
uintptr_t end(const void *pointer,size_t bytes){const auto p=reinterpret_cast<uintptr_t>(pointer);
    if(!p||p%4||bytes>std::numeric_limits<uintptr_t>::max()-p)throw std::invalid_argument("RMSconcat null/alignment/span");return p+bytes;}
bool overlap(const void *a,size_t an,const void *b,size_t bn){return reinterpret_cast<uintptr_t>(a)<end(b,bn)&&reinterpret_cast<uintptr_t>(b)<end(a,an);}
}
RmsConcatResources rmsconcat_resources(){cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,rmsconcat_kernel),"RMSconcat attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,rmsconcat_kernel,1024,0),"RMSconcat occupancy");
    return {a.numRegs,blocks,a.binaryVersion,a.sharedSizeBytes,a.localSizeBytes};
}
void rmsconcat(const float *input,const float *gamma,float *concat,int features,int rows,int offset,float epsilon,cudaStream_t stream){
    if(features!=5120||rows<1||rows>3||(offset!=0&&offset!=features)||!(epsilon>0)||!std::isfinite(epsilon))
        throw std::invalid_argument("RMSconcat fixed MTP geometry/epsilon");
    const size_t input_bytes=size_t(rows)*features*4,output_bytes=2*input_bytes,gamma_bytes=size_t(features)*4;
    end(input,input_bytes);end(gamma,gamma_bytes);end(concat,output_bytes);
    if(overlap(concat,output_bytes,input,input_bytes)||overlap(concat,output_bytes,gamma,gamma_bytes))
        throw std::invalid_argument("RMSconcat input/gamma/output alias");
    rmsconcat_kernel<<<rows,1024,0,stream>>>(input,gamma,concat,nullptr,features,epsilon,features,offset);
    check_cuda(cudaGetLastError(),"RMSconcat direct half-row output");
}
}
