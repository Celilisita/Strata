// Import frozen285 private RN16 input/scalar kernels and original weight oracle.
#define half_row_pair_full_project fast_full_k_frozen285_project
#include "half_pair_full.cu"
#undef half_row_pair_full_project
#include "strata/gemma4/native/fast_full_k_q4.hpp"
namespace strata::gemma4::native {
namespace {
__device__ void coefficient304(uint4 h,int group,float& scale,float& minimum){
    const int shift=(group&3)*8;
    const unsigned a=(h.y>>shift)&255u,b=(h.z>>shift)&255u,c=(h.w>>shift)&255u;
    const int s=group<4?int(a&63u):int((c&15u)|((a>>6)<<4));
    const int m=group<4?int(b&63u):int((c>>4)|((b>>6)<<4));
    scale=__fmul_rn(__half2float(__ushort_as_half(static_cast<unsigned short>(h.x&65535u))),float(s));
    minimum=__fmul_rn(__half2float(__ushort_as_half(static_cast<unsigned short>(h.x>>16))),float(m));
}
__global__ void block_full_weights304(primitive::Matrix gate,primitive::Matrix up,half* output){
    constexpr unsigned blocks_per_matrix=21504u*21u;
    const unsigned block=blockIdx.x*8u+threadIdx.x/16u;
    const unsigned branch=block/blocks_per_matrix,local=block%blocks_per_matrix;
    const unsigned row=local/21u,kb=local%21u,lane=threadIdx.x&15u;
    const auto* raw=static_cast<const unsigned char*>(branch?up.data:gate.data)+size_t(local)*144;
    uint4 header{};if(!lane)header=*reinterpret_cast<const uint4*>(raw);
    const unsigned mask=0xffffu<<((threadIdx.x&31u)&16u);
    const unsigned destination=(branch*21504u+row)*5376u+kb*256u;
#pragma unroll
    for(int part=0;part<4;++part){
        float s0=0,m0=0,s1=0,m1=0;
        if(!lane){coefficient304(header,2*part,s0,m0);coefficient304(header,2*part+1,s1,m1);}
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
FastFullKResources fast_full_k_q4_resources(){cudaFuncAttributes a{};
    primitive::check_cuda(cudaFuncGetAttributes(&a,block_full_weights304),"304 loaded fullK producer attributes");
    return {a.numRegs,int(a.localSizeBytes),int(a.sharedSizeBytes),a.maxThreadsPerBlock};
}
namespace detail304 {
void validate(primitive::Matrix gate,primitive::Matrix up,const FastFullKLease& typed,FastFullKBinding live){
    const auto& lease=typed.operands;const auto& o=lease.output;const auto b=typed.binding;
    constexpr size_t weights=size_t(Rows)*Hidden,inputs=size_t(Tokens)*Hidden;
    constexpr size_t branches=size_t(Tokens)*Rows,raw=weights/256*144;
    if(!live.owner||!live.epoch||!live.stream||b.owner!=live.owner||b.epoch!=live.epoch||b.stream!=live.stream||lease.stream!=live.stream)
        throw std::invalid_argument("Gemma304 owner/epoch/stream");
    for(auto m:{gate,up})if(m.rows!=Rows||m.columns!=Hidden||m.quant!=primitive::Quant::Q4K||m.layout!=primitive::BlockLayout::Gguf)
        throw std::invalid_argument("Gemma304 exact Q4 GateUp5376/21504");
    if(o.tokens!=Tokens||o.rows!=Rows||o.interleaved_values<2*branches||o.activation_values<branches||
       lease.weight_bytes<2*weights*sizeof(half)||lease.input_bytes<inputs*sizeof(half)||
       o.interleaved_values>(std::numeric_limits<size_t>::max)()/4||
       o.activation_values>(std::numeric_limits<size_t>::max)()/4||!lease.handle||lease.workspace_bytes<512)
        throw std::invalid_argument("Gemma304 fixed1280 fullK lease");
    const Span owned[]{{gate.data,raw,16},{up.data,raw,16},{lease.input,inputs*4,4},
        {lease.weight_arena,lease.weight_bytes,16},{lease.input_arena,lease.input_bytes,16},
        {lease.workspace,lease.workspace_bytes,256},{o.interleaved,o.interleaved_values*4,4},
        {o.activation,o.activation_values*4,4}};
    spans(owned,sizeof(owned)/sizeof(owned[0]));
}
void weights(primitive::Matrix gate,primitive::Matrix up,const HalfRowPairLease& p,bool candidate){
    if(candidate)block_full_weights304<<<112896,128,0,p.stream>>>(gate,up,static_cast<half*>(p.weight_arena));
    else encode_weights<<<903168,256,0,p.stream>>>(gate,up,static_cast<half*>(p.weight_arena),size_t(Rows)*Hidden);
    primitive::check_cuda(cudaGetLastError(),"304 full continuous RN16 weights");
}
}
void fast_full_k_q4_project(primitive::Matrix gate,primitive::Matrix up,const FastFullKLease& typed,FastFullKBinding live){
    detail304::validate(gate,up,typed,live);const auto& lease=typed.operands;const auto& o=lease.output;
    constexpr size_t inputs=size_t(Tokens)*Hidden;
    auto* scalar=static_cast<float*>(lease.workspace);
    auto* w=static_cast<half*>(lease.weight_arena);auto* x=static_cast<half*>(lease.input_arena);
    initialize_input_scalars<<<1,1,0,lease.stream>>>(scalar);
    input_maximum<<<unsigned((inputs+255)/256),256,0,lease.stream>>>(lease.input,inputs,scalar);
    choose_input_scale<<<1,1,0,lease.stream>>>(scalar);
    detail304::weights(gate,up,lease,true);
    encode_input<<<unsigned((inputs+255)/256),256,0,lease.stream>>>(lease.input,x,inputs,scalar);
    primitive::check_cuda(cudaGetLastError(),"Gemma304 unchanged197 RN16 pair input preparation");
    blas(cublasSetStream(lease.handle,lease.stream));
    blas(cublasSetWorkspace(lease.handle,static_cast<unsigned char*>(lease.workspace)+256,lease.workspace_bytes-256));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_DEVICE));
    blas(cublasSetMathMode(lease.handle,CUBLAS_TENSOR_OP_MATH));
    blas(cublasGemmEx(lease.handle,CUBLAS_OP_T,CUBLAS_OP_N,2*Rows,Tokens,Hidden,
        scalar+2,w,CUDA_R_16F,Hidden,x,CUDA_R_16F,Hidden,scalar+4,
        o.interleaved,CUDA_R_32F,2*Rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    blas(cublasSetMathMode(lease.handle,CUBLAS_DEFAULT_MATH));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_HOST));
}
}
