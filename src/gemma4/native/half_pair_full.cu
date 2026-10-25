#include "strata/gemma4/native/half_pair_full.hpp"
#include "codec.cuh"
#include "library_input.cuh"
#include <cuda_fp16.h>
#include <limits>
namespace strata::gemma4::native {
namespace {
constexpr int Hidden=5376,Rows=21504,Tokens=1280;
struct Span {const void* data;size_t bytes,alignment;};
bool overlap(Span a,Span b){
    const auto x=reinterpret_cast<std::uintptr_t>(a.data),y=reinterpret_cast<std::uintptr_t>(b.data);
    return x<=y?y-x<a.bytes:x-y<b.bytes;
}
void spans(const Span* s,size_t count){
    for(size_t i=0;i<count;++i){const auto p=reinterpret_cast<std::uintptr_t>(s[i].data);
        if(!p||!s[i].bytes||p%s[i].alignment||s[i].bytes>(std::numeric_limits<std::uintptr_t>::max)()-p)
            throw std::invalid_argument("Gemma285 full advertised span/overflow/alignment");
        for(size_t j=0;j<i;++j)if(overlap(s[i],s[j]))
            throw std::invalid_argument("Gemma285 full source/output/arena alias");
    }
}
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma285 fullK cuBLAS");}
__global__ void encode_weights(primitive::Matrix gate,primitive::Matrix up,
    half* output,size_t weights){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=2*weights)return;
    const bool second=i>=weights;
    const auto matrix=second?up:gate;
    const size_t at=second?i-weights:i;
    const int bytes=matrix.quant==primitive::Quant::Q4K?144:210;
    const auto* raw=static_cast<const unsigned char*>(matrix.data)+(at/256)*bytes;
    output[i]=__float2half_rn(primitive::block_element(raw,matrix.quant,int(at%256)));
}
__global__ void encode_input(const float* input,half* output,size_t count,const float* scalar){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i<count)output[i]=__float2half_rn(input[i]*scalar[1]);
}
}
void half_row_pair_full_project(primitive::Matrix gate,primitive::Matrix up,
    const FullKPairLease& typed,FullKPairBinding live){
    const auto& lease=typed.operands;const auto& o=lease.output;const auto b=typed.binding;
    constexpr size_t weights=size_t(Rows)*Hidden,inputs=size_t(Tokens)*Hidden;
    constexpr size_t branches=size_t(Tokens)*Rows,raw=weights/256*144;
    if(!live.owner||!live.epoch||!live.stream||b.owner!=live.owner||b.epoch!=live.epoch||b.stream!=live.stream||lease.stream!=live.stream)
        throw std::invalid_argument("Gemma285 owner/epoch/stream");
    for(auto m:{gate,up})if(m.rows!=Rows||m.columns!=Hidden||m.quant!=primitive::Quant::Q4K||m.layout!=primitive::BlockLayout::Gguf)
        throw std::invalid_argument("Gemma285 exact Q4 GateUp5376/21504");
    if(o.tokens!=Tokens||o.rows!=Rows||o.interleaved_values<2*branches||o.activation_values<branches||
       lease.weight_bytes<2*weights*sizeof(half)||lease.input_bytes<inputs*sizeof(half)||
       o.interleaved_values>(std::numeric_limits<size_t>::max)()/4||
       o.activation_values>(std::numeric_limits<size_t>::max)()/4||!lease.handle||lease.workspace_bytes<512)
        throw std::invalid_argument("Gemma285 fixed1280 fullK lease");
    const Span owned[]{{gate.data,raw,4},{up.data,raw,4},{lease.input,inputs*4,4},
        {lease.weight_arena,lease.weight_bytes,16},{lease.input_arena,lease.input_bytes,16},
        {lease.workspace,lease.workspace_bytes,256},{o.interleaved,o.interleaved_values*4,4},
        {o.activation,o.activation_values*4,4}};
    spans(owned,sizeof(owned)/sizeof(owned[0]));
    auto* scalar=static_cast<float*>(lease.workspace);
    auto* w=static_cast<half*>(lease.weight_arena);auto* x=static_cast<half*>(lease.input_arena);
    initialize_input_scalars<<<1,1,0,lease.stream>>>(scalar);
    input_maximum<<<unsigned((inputs+255)/256),256,0,lease.stream>>>(lease.input,inputs,scalar);
    choose_input_scale<<<1,1,0,lease.stream>>>(scalar);
    encode_weights<<<unsigned((2*weights+255)/256),256,0,lease.stream>>>(gate,up,w,weights);
    encode_input<<<unsigned((inputs+255)/256),256,0,lease.stream>>>(lease.input,x,inputs,scalar);
    primitive::check_cuda(cudaGetLastError(),"Gemma285 unchanged197 RN16 pair preparation");
    blas(cublasSetStream(lease.handle,lease.stream));
    blas(cublasSetWorkspace(lease.handle,static_cast<unsigned char*>(lease.workspace)+256,lease.workspace_bytes-256));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_DEVICE));
    // FP16 A/B, FP32 compute and C. DISALLOW_REDUCED_PRECISION_REDUCTION is not
    // a remedy for192: C already has the compute precision. No such claim.
    blas(cublasSetMathMode(lease.handle,CUBLAS_TENSOR_OP_MATH));
    blas(cublasGemmEx(lease.handle,CUBLAS_OP_T,CUBLAS_OP_N,2*Rows,Tokens,Hidden,
        scalar+2,w,CUDA_R_16F,Hidden,x,CUDA_R_16F,Hidden,scalar+4,
        o.interleaved,CUDA_R_32F,2*Rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    blas(cublasSetMathMode(lease.handle,CUBLAS_DEFAULT_MATH));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_HOST));
}
}
