#include "p2_vector_encode.hpp"
#include "nvfp4_decode8.hpp"
#include "q8_power2_prefill.cuh"
#include "strata/qwen35/completed_v100/native/nvfp4.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <string>
namespace strata::completed_v100::qwen35::native {
namespace {
void checked(cublasStatus_t status,const char* op) {
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(op)+": "+std::to_string(int(status)));
}
__global__ void encode(const float* input,half* output,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const float value=input[i];float maximum=fabsf(value);
    for(int offset=8;offset;offset/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset,16));
    const unsigned invalid=__ballot_sync(0xffffffff,!isfinite(value)||fabsf(value)>65024.f);
    const bool bad=(invalid&(0xffffu<<(threadIdx.x&16)))!=0;
    int exponent=0;const float fraction=frexpf(maximum/127.f,&exponent);
    const int power=max(-14,min(9,exponent-(fraction==.5f)));
    const float scale=maximum>0.f?ldexpf(1.f,power):ldexpf(1.f,-14);
    const int code=max(-127,min(127,__float2int_rn(value/scale)));
    output[i]=__float2half_rn(bad?CUDART_NAN_F:float(code)*scale);
}
__global__ void reconstruct(const half* input,float* output,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)output[i]=__half2float(input[i]);
}
}
void q8p2_encode(const float* input,void* output,size_t count,cudaStream_t stream) {
    if(!input||!output||static_cast<const void*>(input)==output||!count||count%64)
        throw std::invalid_argument("Q8-P2 encoder extent/alias");
    encode<<<unsigned((count+127)/128),128,0,stream>>>(input,static_cast<half*>(output),count);
    check_cuda(cudaGetLastError(),"Q8-P2 group16 activation");
}
void q8p2_reconstruct(const void* input,float* output,size_t count,cudaStream_t stream) {
    if(!input||!output||input==static_cast<void*>(output)||!count||count%64)
        throw std::invalid_argument("Q8-P2 reconstruct extent/alias");
    reconstruct<<<unsigned((count+255)/256),256,0,stream>>>(static_cast<const half*>(input),output,count);
    check_cuda(cudaGetLastError(),"Q8-P2 actual FP16 operand reconstruction");
}
void q8p2_nvfp4_prefill(const void* raw,const float* scale,const float* input,float* output,
    int rows,int columns,int tokens,Q8P2Workspace w,cudaStream_t stream) {
    if(!raw||!scale||!input||!output||input==output||rows<1||columns<64||columns%64||tokens<4||tokens>8192||
       size_t(rows)*columns>w.weight_capacity||size_t(tokens)*columns>w.input_capacity||
       !w.half_weights||!w.half_input||!w.handle||!w.vendor||!w.vendor_bytes||!w.zero)
        throw std::invalid_argument("Q8-P2 PP owned arena/extent");
    if(w.decode8)nvfp4_decode_raw_half8(raw,w.half_weights,rows,columns,stream);
    else nvfp4_decode_raw_half(raw,w.half_weights,rows,columns,stream);
    if(w.vector_encode)q8p2_encode_vector(input,w.half_input,size_t(tokens)*columns,stream);
    else q8p2_encode(input,w.half_input,size_t(tokens)*columns,stream);
    checked(cublasSetStream(w.handle,stream),"Q8-P2 cuBLAS stream");
    checked(cublasSetWorkspace(w.handle,w.vendor,w.vendor_bytes),"Q8-P2 stable cuBLAS workspace");
    checked(cublasSetPointerMode(w.handle,CUBLAS_POINTER_MODE_DEVICE),"Q8-P2 device coefficients");
    checked(cublasGemmEx(w.handle,CUBLAS_OP_T,CUBLAS_OP_N,rows,tokens,columns,scale,
        w.half_weights,CUDA_R_16F,columns,w.half_input,CUDA_R_16F,columns,w.zero,
        output,CUDA_R_32F,rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"Q8-P2 single Tensor Core PP GEMM");
}
}
