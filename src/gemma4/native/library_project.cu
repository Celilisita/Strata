#include "strata/gemma4/native/ffn.hpp"
#include "codec.cuh"
#include <cuda_fp16.h>
#include "library_input.cuh"
namespace strata::gemma4::native {
namespace {
// A = A0 + A1/4096 + residual. Scaling the correction before FP16 storage
// avoids losing small weight residuals to half subnormal rounding.
__device__ void encode(float value,half* first,half* correction){
    const half main=__float2half_rn(value);
    *first=main;*correction=__float2half_rn((value-__half2float(main))*4096.f);
}
__global__ void weight_planes(primitive::Matrix m,half* output,size_t count){
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;if(index>=count)return;
    const int bytes=m.layout==primitive::BlockLayout::Q6Aligned224?224:
        m.quant==primitive::Quant::Q4K?144:210;
    const auto* raw=static_cast<const unsigned char*>(m.data)+(index/256)*bytes;
    encode(primitive::block_element(raw,m.quant,int(index%256)),output+index,output+count+index);
}
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma residual cuBLAS projection");}
}
void residual_library_project(primitive::Matrix m,const float* input,float* output,int tokens,
    void* weight_arena,void* input_arena,cublasHandle_t handle,void* workspace,cudaStream_t stream,int base_slice,int algorithm){
    q6_down_weight_bytes(m);
    if(algorithm< -1||algorithm>1||!m.data||!input||!output||!weight_arena||!input_arena||input_arena==input||input_arena==output||
        m.rows<1||m.columns<256||m.columns%256||tokens<128||tokens>8192||(base_slice!=0&&base_slice!=512&&base_slice!=896&&base_slice!=1024)||
        (m.quant!=primitive::Quant::Q4K&&m.quant!=primitive::Quant::Q6K))throw std::invalid_argument("Gemma residual cuBLAS operand lease");
    const size_t weights=size_t(m.rows)*m.columns,values=size_t(tokens)*m.columns;
    auto* w=static_cast<half*>(weight_arena);auto* x=static_cast<half*>(input_arena);
    auto* scalars=static_cast<float*>(workspace);
    initialize_input_scalars<<<1,1,0,stream>>>(scalars);
    input_maximum<<<unsigned((values+255)/256),256,0,stream>>>(input,values,scalars);
    choose_input_scale<<<1,1,0,stream>>>(scalars);
    weight_planes<<<unsigned((weights+255)/256),256,0,stream>>>(m,w,weights);
    input_planes_scaled<<<unsigned((values+255)/256),256,0,stream>>>(input,x,values,scalars);
    primitive::check_cuda(cudaGetLastError(),"Gemma two scaled half operand planes");
    blas(cublasSetStream(handle,stream));blas(cublasSetWorkspace(handle,static_cast<unsigned char*>(workspace)+256,32u*1024*1024-256));
    blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE));blas(cublasSetMathMode(handle,static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const float *one=scalars+5,*zero=scalars+4,*alpha=scalars+2,*correction=scalars+3;
    auto gemm=[&](const half* a,const half* b,const float* alpha,const float* beta,int reduction){
        blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,m.rows,tokens,reduction,alpha,
            a,CUDA_R_16F,m.columns,b,CUDA_R_16F,m.columns,beta,output,CUDA_R_32F,m.rows,
            CUBLAS_COMPUTE_32F,algorithm<0?CUBLAS_GEMM_DEFAULT_TENSOR_OP:algorithm==0?CUBLAS_GEMM_ALGO0_TENSOR_OP:CUBLAS_GEMM_ALGO1_TENSOR_OP));
    };
    // Bound main-product accumulation length; retain F32 output between slices.
    // Correction products are much smaller and keep a single full-K submission.
    const int slice=base_slice?base_slice:m.columns;
    for(int first=0;first<m.columns;first+=slice){
        const int count=(m.columns-first<slice)?m.columns-first:slice;
        gemm(w+first,x+first,alpha,first?one:zero,count);
    }
    gemm(w,x+values,correction,one,m.columns);
    gemm(w+weights,x,correction,one,m.columns);
    blas(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
    blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
    // Power-of-two input scaling is undone by GPU-owned alpha values.
    // The A1*B1 cross term and second-order representation tails are the only
    // deliberate approximation. Real-weight <=1e-5 gates precede admission.
}
}
