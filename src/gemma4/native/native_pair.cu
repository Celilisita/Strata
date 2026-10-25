#include "strata/gemma4/native/ffn.hpp"
#include "codec.cuh"
#include "library_input.cuh"
#include <cuda_fp16.h>
namespace strata::gemma4::native {
namespace {
// A=[W_high,W_low], B=[X_low,X_high]. One correction GEMM computes
// W_high*X_low + W_low*X_high, while the main GEMM reads W_high/X_high.
// Low planes are scaled by4096, as in the admitted residual library path.
__global__ void joint_weights(primitive::Matrix matrix,half* output,size_t count){
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if(index>=count)return;
    const auto* raw=static_cast<const unsigned char*>(matrix.data)+(index/256)*144;
    const float value=primitive::block_element(raw,matrix.quant,int(index%256));
    const half high=__float2half_rn(value);
    const size_t at=(index/matrix.columns)*(2*matrix.columns)+index%matrix.columns;
    output[at]=high;
    output[at+matrix.columns]=__float2half_rn((value-__half2float(high))*4096.f);
}
__global__ void joint_input(const float* source,half* output,size_t count,int columns,const float* scalar){
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if(index>=count)return;
    const float value=source[index]*scalar[1];const half high=__float2half_rn(value);
    const size_t at=(index/columns)*(2*columns)+index%columns;
    output[at]=__float2half_rn((value-__half2float(high))*4096.f);
    output[at+columns]=high;
}
void blas(cublasStatus_t status){
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma joint residual cuBLAS projection");
}
}
void configure_native_pair(){} // No attributes, GPU allocation, or library work in this cold selector.
void native_pair_project(primitive::Matrix gate,primitive::Matrix up,const PairPrefillLease& lease){
    const size_t weights=size_t(gate.rows)*gate.columns,values=size_t(lease.tokens)*gate.columns;
    if(!gate.data||!up.data||gate.quant!=primitive::Quant::Q4K||up.quant!=primitive::Quant::Q4K||
       gate.layout!=primitive::BlockLayout::Gguf||up.layout!=primitive::BlockLayout::Gguf||
       gate.rows<1||gate.rows!=up.rows||gate.columns!=up.columns||gate.columns<256||gate.columns%256||
       !lease.owner||!lease.epoch||!lease.source||!lease.gate||!lease.up||lease.gate==lease.up||
       !lease.weight_arena||!lease.input_arena||!lease.handle||!lease.workspace||!lease.stream||
       lease.input_arena==lease.source||lease.input_arena==lease.gate||lease.input_arena==lease.up||
       lease.tokens<1280||lease.tokens>8192||weights*4>lease.weight_bytes||values*4>lease.input_bytes||
       lease.output_stride<size_t(lease.tokens)*gate.rows||
       (lease.slice!=512&&lease.slice!=896&&lease.slice!=1024))
        throw std::invalid_argument("Gemma joint residual source/shape/precision/epoch lease");
    auto* scalar=static_cast<float*>(lease.workspace);
    auto* weight=static_cast<half*>(lease.weight_arena);auto* input=static_cast<half*>(lease.input_arena);
    initialize_input_scalars<<<1,1,0,lease.stream>>>(scalar);
    input_maximum<<<unsigned((values+255)/256),256,0,lease.stream>>>(lease.source,values,scalar);
    choose_input_scale<<<1,1,0,lease.stream>>>(scalar);
    joint_input<<<unsigned((values+255)/256),256,0,lease.stream>>>(lease.source,input,values,gate.columns,scalar);
    primitive::check_cuda(cudaGetLastError(),"Gemma joint residual input planes");
    blas(cublasSetStream(lease.handle,lease.stream));
    blas(cublasSetWorkspace(lease.handle,static_cast<unsigned char*>(lease.workspace)+256,32u*1024*1024-256));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_DEVICE));
    blas(cublasSetMathMode(lease.handle,static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const float* one=scalar+5,*zero=scalar+4,*alpha=scalar+2,*correction=scalar+3;
    for(int branch=0;branch<2;++branch){
        const auto matrix=branch?up:gate;float* output=branch?lease.up:lease.gate;
        joint_weights<<<unsigned((weights+255)/256),256,0,lease.stream>>>(matrix,weight,weights);
        primitive::check_cuda(cudaGetLastError(),"Gemma joint residual weight planes");
        auto gemm=[&](const half* a,const half* b,const float* scale,const float* beta,int reduction){
            blas(cublasGemmEx(lease.handle,CUBLAS_OP_T,CUBLAS_OP_N,matrix.rows,lease.tokens,reduction,scale,
                a,CUDA_R_16F,2*matrix.columns,b,CUDA_R_16F,2*matrix.columns,beta,output,CUDA_R_32F,matrix.rows,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        for(int first=0;first<matrix.columns;first+=lease.slice){
            const int count=matrix.columns-first<lease.slice?matrix.columns-first:lease.slice;
            gemm(weight+first,input+matrix.columns+first,alpha,first?one:zero,count);
        }
        gemm(weight,input,correction,one,2*matrix.columns);
    }
    blas(cublasSetMathMode(lease.handle,CUBLAS_DEFAULT_MATH));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_HOST));
}
}
