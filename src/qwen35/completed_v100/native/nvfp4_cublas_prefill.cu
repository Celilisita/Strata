#include "strata/qwen35/completed_v100/native/nvfp4_projection.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <algorithm>
#include <string>
#include <stdexcept>
namespace strata::completed_v100::qwen35::native {
namespace {
void checked_pp(cublasStatus_t status,const char* operation) {
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(operation)+": "+std::to_string(int(status)));
}
__global__ void pack_combined(const float* input,half* combined,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) {
        const float value=input[i];const half high=__float2half_rn(value);
        combined[i]=high;combined[count+i]=__float2half_rn((value-__half2float(high))*4096.f);
    }
}
__global__ void reduce_combined(const float* partial,float* output,const float* scale,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)output[i]=fmaf(partial[count+i],1.f/4096.f,partial[i])*scale[0];
}
}
void Nvfp4Projection::run_combined_prefill(const void* blocks,const float* scale,const float* input,float* output,
    int rows,int columns,int tokens,cudaStream_t stream) {
    if(size_t(rows)*columns>weight_values_||columns>input_width_||tokens<1||tokens>maximum_tokens_)
        throw std::invalid_argument("Combined cuBLAS PP exceeds owned arenas");
    const size_t capacity=weight_values_/2;
    const int chunk_limit=int((std::min)(size_t(maximum_tokens_),capacity/(size_t(2)*rows)));
    if(chunk_limit<1)throw std::invalid_argument("Combined cuBLAS PP partial arena too small");
    // The first half of the existing arena holds decoded weights. Its second
    // half is disjoint F32 output scratch; no additional device allocation.
    auto* partial=reinterpret_cast<float*>(static_cast<half*>(weights_)+weight_values_);
    nvfp4_decode_raw_half(blocks,weights_,rows,columns,stream);
    checked_pp(cublasSetStream(handle_,stream),"Combined cuBLAS PP stream");
    checked_pp(cublasSetWorkspace(handle_,vendor_workspace_,VendorWorkspaceBytes),"Combined cuBLAS PP workspace");
    checked_pp(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_DEVICE),"Combined cuBLAS PP coefficients");
    for(int first=0;first<tokens;first+=chunk_limit) {
        const int count=(std::min)(tokens-first,chunk_limit);
        const size_t input_count=size_t(count)*columns,output_count=size_t(count)*rows;
        pack_combined<<<unsigned((input_count+255)/256),256,0,stream>>>(input+size_t(first)*columns,static_cast<half*>(high_),input_count);
        check_cuda(cudaGetLastError(),"Combined cuBLAS PP high/low input");
        checked_pp(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,rows,2*count,columns,constants_,
            weights_,CUDA_R_16F,columns,high_,CUDA_R_16F,columns,constants_+3,partial,CUDA_R_32F,rows,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"Combined NVIDIA cuBLAS high/low PP");
        reduce_combined<<<unsigned((output_count+255)/256),256,0,stream>>>(partial,output+size_t(first)*rows,scale,output_count);
        check_cuda(cudaGetLastError(),"Combined cuBLAS PP complete output");
    }
}
}
