#include "strata/gemma4/native/ffn.hpp"
#include "codec.cuh"
#include "library_input.cuh"
#include <cuda_fp16.h>
#include <climits>
namespace strata::gemma4::native {
namespace {
__global__ void half_weight(primitive::Matrix matrix,half* output,size_t count){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=count)return;
    const int bytes=matrix.layout==primitive::BlockLayout::Q6Aligned224?224:
        matrix.quant==primitive::Quant::Q4K?144:210;
    const auto* raw=static_cast<const unsigned char*>(matrix.data)+(i/256)*bytes;
    output[i]=__float2half_rn(primitive::block_element(raw,matrix.quant,int(i%256)));
}
__global__ void half_input(const float* input,half* output,size_t count,const float* scalar){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i<count)output[i]=__float2half_rn(input[i]*scalar[1]);
}
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma FP16 candidate cuBLAS projection");}
}
void prepare_half_library_weights(primitive::Matrix matrix,void* output,size_t capacity,cudaStream_t stream){
    const size_t raw_bytes=q6_down_weight_bytes(matrix);
    const size_t values=size_t(matrix.rows)*matrix.columns,bytes=values*sizeof(half);
    const auto source=reinterpret_cast<std::uintptr_t>(matrix.data),target=reinterpret_cast<std::uintptr_t>(output);
    if(!output||target%alignof(half)||!stream||values>size_t(INT_MAX)||capacity<bytes||
       (source<=target?target-source<raw_bytes:source-target<bytes))
        throw std::invalid_argument("Gemma RN16 weight reader output extent/lifetime");
    half_weight<<<unsigned((values+255)/256),256,0,stream>>>(matrix,static_cast<half*>(output),values);
    primitive::check_cuda(cudaGetLastError(),"Gemma RN16 original or aligned-Q6 weight reader");
}
void half_library_project(primitive::Matrix matrix,const float* input,float* output,int tokens,
    void* weight_arena,void* input_arena,cublasHandle_t handle,void* workspace,cudaStream_t stream,int slice){
    q6_down_weight_bytes(matrix);
    if(!matrix.data||!input||!output||!weight_arena||!input_arena||input_arena==input||input_arena==output||
       !handle||!workspace||!stream||matrix.rows<1||matrix.columns<256||matrix.columns%256||
       tokens<128||tokens>8192||(slice!=0&&slice!=512&&slice!=896&&slice!=1024)||
       (matrix.quant!=primitive::Quant::Q4K&&matrix.quant!=primitive::Quant::Q6K))
        throw std::invalid_argument("Gemma FP16 candidate operand lease");
    const size_t weights=size_t(matrix.rows)*matrix.columns,values=size_t(tokens)*matrix.columns;
    auto* w=static_cast<half*>(weight_arena);auto* x=static_cast<half*>(input_arena);auto* scalar=static_cast<float*>(workspace);
    initialize_input_scalars<<<1,1,0,stream>>>(scalar);
    input_maximum<<<unsigned((values+255)/256),256,0,stream>>>(input,values,scalar);
    choose_input_scale<<<1,1,0,stream>>>(scalar);
    prepare_half_library_weights(matrix,w,weights*sizeof(half),stream);
    half_input<<<unsigned((values+255)/256),256,0,stream>>>(input,x,values,scalar);
    primitive::check_cuda(cudaGetLastError(),"Gemma FP16 candidate encoding");
    blas(cublasSetStream(handle,stream));blas(cublasSetWorkspace(handle,static_cast<unsigned char*>(workspace)+256,32u*1024*1024-256));
    blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE));
    blas(cublasSetMathMode(handle,static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const int step=slice?slice:matrix.columns;
    for(int first=0;first<matrix.columns;first+=step){
        const int count=matrix.columns-first<step?matrix.columns-first:step;
        blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,matrix.rows,tokens,count,scalar+2,
            w+first,CUDA_R_16F,matrix.columns,x+first,CUDA_R_16F,matrix.columns,scalar+(first?5:4),output,CUDA_R_32F,matrix.rows,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    blas(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
}
}
