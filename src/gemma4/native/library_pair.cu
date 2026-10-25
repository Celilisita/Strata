#include "strata/gemma4/native/ffn.hpp"
#include "codec.cuh"
#include "library_input.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <string>
namespace strata::gemma4::native {
namespace {
enum class InputType {ScaledHalfHiLow};
struct PreparedInput {
    const float* source;half* planes;float* scalars;
    int columns,tokens;const void* owner;std::uint64_t epoch;
    cudaStream_t stream;InputType type=InputType::ScaledHalfHiLow;
};
__global__ void paired_planes(primitive::Matrix gate,primitive::Matrix up,half* out,int first,int rows){
    const auto m=blockIdx.z?up:gate;const int blocks=m.columns/256;
    const size_t per=size_t(rows)*m.columns;
    const size_t index=size_t(blockIdx.z)*per+(size_t(blockIdx.y)*blocks+blockIdx.x)*256+threadIdx.x;
    const int bytes=m.quant==primitive::Quant::Q4K?144:210;
    const auto* raw=static_cast<const unsigned char*>(m.data)+(size_t(first+blockIdx.y)*blocks+blockIdx.x)*bytes;
    const float value=primitive::block_element(raw,m.quant,int(threadIdx.x));const half main=__float2half_rn(value);
    out[index]=main;out[per*2+index]=__float2half_rn((value-__half2float(main))*4096.f);
}
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma paired GPU cuBLAS status "+std::to_string(int(status)));}
bool matrix_valid(primitive::Matrix m){return m.data&&m.rows>=256&&m.columns>=256&&m.columns%256==0&&
    m.layout==primitive::BlockLayout::Gguf&&(m.quant==primitive::Quant::Q4K||m.quant==primitive::Quant::Q6K);}
PreparedInput prepare(primitive::Matrix m,const PairPrefillLease& x){
    const size_t values=size_t(x.tokens)*m.columns;
    if(!x.owner||!x.epoch||!x.source||!x.input_arena||!x.workspace||!x.stream||values*4>x.input_bytes||x.input_arena==x.source)
        throw std::invalid_argument("Gemma paired input source/dtype/owner/epoch lease");
    auto* scalars=static_cast<float*>(x.workspace);auto* planes=static_cast<half*>(x.input_arena);
    initialize_input_scalars<<<1,1,0,x.stream>>>(scalars);
    input_maximum<<<unsigned((values+255)/256),256,0,x.stream>>>(x.source,values,scalars);
    choose_input_scale<<<1,1,0,x.stream>>>(scalars);
    input_planes_scaled<<<unsigned((values+255)/256),256,0,x.stream>>>(x.source,planes,values,scalars);
    primitive::check_cuda(cudaGetLastError(),"Gemma shared GateUp rangeguard andhalfplanes");
    return {x.source,planes,scalars,m.columns,x.tokens,x.owner,x.epoch,x.stream};
}
}
void paired_library_project(primitive::Matrix gate,primitive::Matrix up,const PairPrefillLease& x){
    if(!matrix_valid(gate)||!matrix_valid(up)||gate.rows!=up.rows||gate.columns!=up.columns||!x.gate||!x.up||
        !x.weight_arena||!x.handle||x.tokens<1280||x.tokens>8192||x.output_stride<size_t(x.tokens)*gate.rows||
        reinterpret_cast<std::uintptr_t>(x.up)-reinterpret_cast<std::uintptr_t>(x.gate)!=x.output_stride*sizeof(float)||
        (x.slice!=0&&x.slice!=512&&x.slice!=896&&x.slice!=1024)||x.algorithm< -1||x.algorithm>1)
        throw std::invalid_argument("Gemma paired rawweight/output arena lease");
    const int tile=int((std::min)(size_t(65408),x.weight_bytes/(size_t(gate.columns)*8))/128)*128;
    if(tile<128)throw std::invalid_argument("Gemma paired decoded arena extent");
    const auto input=prepare(gate,x);
    if(input.source!=x.source||input.columns!=gate.columns||input.tokens!=x.tokens||input.owner!=x.owner||
       input.epoch!=x.epoch||input.stream!=x.stream||input.type!=InputType::ScaledHalfHiLow)
        throw std::invalid_argument("Gemma paired prepared input identity");
    blas(cublasSetStream(x.handle,x.stream));
    blas(cublasSetWorkspace(x.handle,static_cast<unsigned char*>(x.workspace)+256,32u*1024*1024-256));
    blas(cublasSetPointerMode(x.handle,CUBLAS_POINTER_MODE_DEVICE));
    blas(cublasSetMathMode(x.handle,static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    auto* weight=static_cast<half*>(x.weight_arena);const size_t values=size_t(x.tokens)*gate.columns;
    const auto algo=x.algorithm<0?CUBLAS_GEMM_DEFAULT_TENSOR_OP:x.algorithm==0?CUBLAS_GEMM_ALGO0_TENSOR_OP:CUBLAS_GEMM_ALGO1_TENSOR_OP;
    for(int row=0;row<gate.rows;row+=tile){
        const int rows=(std::min)(tile,gate.rows-row);const size_t weights=size_t(rows)*gate.columns;
        paired_planes<<<dim3(gate.columns/256,rows,2),256,0,x.stream>>>(gate,up,weight,row,rows);
        primitive::check_cuda(cudaGetLastError(),"Gemma bounded paired weight planes");
        auto gemm=[&](const half* a,const half* b,const float* alpha,const float* beta,int reduction){
            blas(cublasGemmStridedBatchedEx(x.handle,CUBLAS_OP_T,CUBLAS_OP_N,rows,x.tokens,reduction,alpha,
                a,CUDA_R_16F,gate.columns,static_cast<long long>(weights),b,CUDA_R_16F,gate.columns,0,
                beta,x.gate+row,CUDA_R_32F,gate.rows,static_cast<long long>(x.output_stride),2,CUBLAS_COMPUTE_32F,algo));
        };
        const int slice=x.slice?x.slice:gate.columns;
        for(int first=0;first<gate.columns;first+=slice){const int count=(std::min)(slice,gate.columns-first);
            gemm(weight+first,input.planes+first,input.scalars+2,input.scalars+(first?5:4),count);}
        gemm(weight,input.planes+values,input.scalars+3,input.scalars+5,gate.columns);
        gemm(weight+weights*2,input.planes,input.scalars+3,input.scalars+5,gate.columns);
    }
    blas(cublasSetMathMode(x.handle,CUBLAS_DEFAULT_MATH));blas(cublasSetPointerMode(x.handle,CUBLAS_POINTER_MODE_HOST));
    // GELU is submitted by Ffn only after both branches and all correction
    // products are complete. Source/arena lifetime ends at this stream boundary.
}
}
