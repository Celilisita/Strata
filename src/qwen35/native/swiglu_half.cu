#include "strata/qwen35/native/dense_gemm.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "math.cuh"
#include <cuda_fp16.h>

namespace strata::qwen35::native {
namespace {
__global__ void activate_half(const float *gate,const float *up,half *output,size_t n){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)output[i]=__float2half_rn(swiglu_value(gate[i],up[i]));
}
}
void swiglu_half(const float *gate,const float *up,void *output,size_t n,cudaStream_t stream){
    activate_half<<<unsigned((n+255)/256),256,0,stream>>>(gate,up,static_cast<half *>(output),n);
    check_cuda(cudaGetLastError(),"Strata fused SwiGLU half output");
}
}
