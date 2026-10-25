// Frozen227 input encoding/products remain unchanged; the public wrapper is renamed.
#define half_row_pair_three_panel_project pp3_q4_frozen_parent_project
#include "half_k_three_panels.cu"
#undef half_row_pair_three_panel_project
#include "strata/gemma4/native/pp3_q4_panel.hpp"
namespace strata::gemma4::native {
namespace {
__device__ void coefficient(uint4 h,int group,float& scale,float& minimum){
    const int shift=(group&3)*8;
    const unsigned a=(h.y>>shift)&255u,b=(h.z>>shift)&255u,c=(h.w>>shift)&255u;
    const int s=group<4?int(a&63u):int((c&15u)|((a>>6)<<4));
    const int m=group<4?int(b&63u):int((c>>4)|((b>>6)<<4));
    scale=__fmul_rn(__half2float(__ushort_as_half(static_cast<unsigned short>(h.x&65535u))),float(s));
    minimum=__fmul_rn(__half2float(__ushort_as_half(static_cast<unsigned short>(h.x>>16))),float(m));
}
__global__ void block_panel_weights(primitive::Matrix gate,primitive::Matrix up,half* output){
    constexpr unsigned blocks_per_matrix=21504u*21u;
    const unsigned block=blockIdx.x*8u+threadIdx.x/16u;
    const unsigned branch=block/blocks_per_matrix,local=block%blocks_per_matrix;
    const unsigned row=local/21u,kb=local%21u,lane=threadIdx.x&15u;
    const auto* raw=static_cast<const unsigned char*>(branch?up.data:gate.data)+size_t(local)*144;
    uint4 header{};if(!lane)header=*reinterpret_cast<const uint4*>(raw);
    const unsigned mask=0xffffu<<((threadIdx.x&31u)&16u);
    const unsigned destination=(kb/7u)*(43008u*1792u)+(branch*21504u+row)*1792u+(kb%7u)*256u;
#pragma unroll
    for(int part=0;part<4;++part){
        float s0=0,m0=0,s1=0,m1=0;
        if(!lane){coefficient(header,2*part,s0,m0);coefficient(header,2*part+1,s1,m1);}
        s0=__shfl_sync(mask,s0,0,16);m0=__shfl_sync(mask,m0,0,16);
        s1=__shfl_sync(mask,s1,0,16);m1=__shfl_sync(mask,m1,0,16);
        const unsigned q=*reinterpret_cast<const unsigned short*>(raw+16+part*32+lane*2);
        const half low0=__float2half_rn(__fsub_rn(__fmul_rn(s0,float(q&15u)),m0));
        const half low1=__float2half_rn(__fsub_rn(__fmul_rn(s0,float((q>>8)&15u)),m0));
        const half high0=__float2half_rn(__fsub_rn(__fmul_rn(s1,float((q>>4)&15u)),m1));
        const half high1=__float2half_rn(__fsub_rn(__fmul_rn(s1,float((q>>12)&15u)),m1));
        *reinterpret_cast<half2*>(output+destination+part*64+lane*2)=__halves2half2(low0,low1);
        *reinterpret_cast<half2*>(output+destination+part*64+lane*2+32)=__halves2half2(high0,high1);
    }
}
}
Q4PpPanelResources q4_pp_panel_resources(){cudaFuncAttributes a{};
    primitive::check_cuda(cudaFuncGetAttributes(&a,block_panel_weights),"298 loaded panel kernel attributes");
    return {a.numRegs,int(a.localSizeBytes),int(a.sharedSizeBytes),a.maxThreadsPerBlock};
}
namespace detail298 {
void scale(const HalfRowPairLease& p){auto* scalar=static_cast<float*>(p.workspace);
    initialize_input_scalars<<<1,1,0,p.stream>>>(scalar);
    input_maximum<<<unsigned((size_t(1280)*5376+255)/256),256,0,p.stream>>>(p.input,size_t(1280)*5376,scalar);
    choose_input_scale<<<1,1,0,p.stream>>>(scalar);
    primitive::check_cuda(cudaGetLastError(),"298 unchanged input range/scale");
}
void weights(primitive::Matrix gate,primitive::Matrix up,const HalfRowPairLease& p,bool candidate){
    if(candidate)block_panel_weights<<<112896,128,0,p.stream>>>(gate,up,static_cast<half*>(p.weight_arena));
    else panel_weights<<<903168,256,0,p.stream>>>(gate,up,static_cast<half*>(p.weight_arena));
    primitive::check_cuda(cudaGetLastError(),"298 full RN16 weight panels");
}
void input(const HalfRowPairLease& p){
    panel_input<<<unsigned((size_t(1280)*5376+255)/256),256,0,p.stream>>>(p.input,static_cast<half*>(p.input_arena),static_cast<float*>(p.workspace));
    primitive::check_cuda(cudaGetLastError(),"298 unchanged input panel packing");
}
void products(const HalfRowPairLease& p){
    auto* scalar=static_cast<float*>(p.workspace);auto* w=static_cast<half*>(p.weight_arena);auto* x=static_cast<half*>(p.input_arena);
    blas(cublasSetStream(p.handle,p.stream));
    blas(cublasSetWorkspace(p.handle,static_cast<unsigned char*>(p.workspace)+256,p.workspace_bytes-256));
    blas(cublasSetPointerMode(p.handle,CUBLAS_POINTER_MODE_DEVICE));
    blas(cublasSetMathMode(p.handle,static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    for(int j=0;j<3;++j)blas(cublasGemmEx(p.handle,CUBLAS_OP_T,CUBLAS_OP_N,43008,1280,1792,scalar+2,
        w+size_t(j)*43008*1792,CUDA_R_16F,1792,x+size_t(j)*1280*1792,CUDA_R_16F,1792,
        scalar+(j?5:4),p.output.interleaved,CUDA_R_32F,43008,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    blas(cublasSetMathMode(p.handle,CUBLAS_DEFAULT_MATH));blas(cublasSetPointerMode(p.handle,CUBLAS_POINTER_MODE_HOST));
}
}
void q4_pp_panel_project(primitive::Matrix gate,primitive::Matrix up,const Q4PpPanelLease& p,Q4PpPanelBinding live){
    detail298::validate(gate,up,p,live);detail298::scale(p.operands);
    detail298::weights(gate,up,p.operands,true);detail298::input(p.operands);detail298::products(p.operands);
}
}
