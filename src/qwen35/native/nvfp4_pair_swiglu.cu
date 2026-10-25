#include "nvfp4_prefill_pair.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "math.cuh"
namespace strata::qwen35::native {
namespace {
__global__ void epilogue(const float* x,const float* sg,const float* su,
    float* activation,int rows,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const size_t at=(i/rows)*size_t(2*rows)+i%rows;
    const float g=__fmul_rn(x[at],sg[0]),u=__fmul_rn(x[at+rows],su[0]);
    activation[i]=swiglu_value(g,u);
}
}
void nvfp4_pair_swiglu(const float* x,const float* sg,const float* su,float* a,int rows,int tokens,cudaStream_t s) {
    if(!x||!sg||!su||!a||rows<1||tokens<1||x==a)throw std::invalid_argument("NVFP4pairedSwiGLUlease");
    const size_t count=size_t(rows)*tokens;
    epilogue<<<unsigned((count+255)/256),256,0,s>>>(x,sg,su,a,rows,count);
    check_cuda(cudaGetLastError(),"NVFP4scaledpairedSwiGLU");
}
}
