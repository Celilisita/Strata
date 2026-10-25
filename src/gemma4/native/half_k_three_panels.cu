#include "half_k_three_panels.hpp"
#include "codec.cuh"
#include "library_input.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::gemma4::native {
namespace {
constexpr int kTokens=1280,kHidden=5376,kRows=21504,kPanel=1792,kSegments=3;
constexpr size_t kMergedRows=size_t(2)*kRows;
static_assert(kPanel*kSegments==kHidden,"227 exact K partition");
bool overlaps(const void* a,size_t a_bytes,const void* b,size_t b_bytes){
    const auto first=reinterpret_cast<std::uintptr_t>(a),second=reinterpret_cast<std::uintptr_t>(b);
    return first<=second?second-first<a_bytes:first-second<b_bytes;
}
bool matrix_valid(primitive::Matrix matrix){
    return matrix.data&&matrix.rows==kRows&&matrix.columns==kHidden&&
        matrix.layout==primitive::BlockLayout::Gguf&&
        (matrix.quant==primitive::Quant::Q4K||matrix.quant==primitive::Quant::Q6K);
}
void blas(cublasStatus_t status){
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma compact K1792 cuBLAS projection");
}
__global__ void panel_weights(primitive::Matrix gate,primitive::Matrix up,half* output){
    const size_t destination=size_t(blockIdx.x)*256+threadIdx.x;
    constexpr size_t panel_values=kMergedRows*kPanel,total=panel_values*kSegments;
    if(destination>=total)return;
    const int segment=int(destination/panel_values);
    const size_t local=destination%panel_values;
    const int merged_row=int(local/kPanel),column=segment*kPanel+int(local%kPanel);
    const bool second=merged_row>=kRows;
    const auto matrix=second?up:gate;
    const size_t source=size_t(second?merged_row-kRows:merged_row)*kHidden+column;
    const int bytes=matrix.quant==primitive::Quant::Q4K?144:210;
    const auto* raw=static_cast<const unsigned char*>(matrix.data)+(source/256)*bytes;
    //1792 is seven original 256-value blocks. Preserve absolute source
    //indexing so Q4/Q6 metadata and the Gate/Up boundary are unchanged.
    output[destination]=__float2half_rn(primitive::block_element(raw,matrix.quant,int(source%256)));
}
__global__ void panel_input(const float* input,half* output,const float* scalar){
    const size_t destination=size_t(blockIdx.x)*256+threadIdx.x;
    constexpr size_t panel_values=size_t(kTokens)*kPanel,total=panel_values*kSegments;
    if(destination>=total)return;
    const int segment=int(destination/panel_values);
    const size_t local=destination%panel_values;
    const size_t source=(local/kPanel)*kHidden+segment*kPanel+local%kPanel;
    output[destination]=__float2half_rn(input[source]*scalar[1]);
}
}

void half_row_pair_three_panel_project(primitive::Matrix gate,primitive::Matrix up,
    const HalfRowPairLease& lease){
    constexpr size_t inputs=size_t(kTokens)*kHidden,branch_values=size_t(kTokens)*kRows;
    constexpr size_t weight_bytes=kMergedRows*kHidden*sizeof(half),input_bytes=inputs*sizeof(half);
    const auto& out=lease.output;
    if(!matrix_valid(gate)||!matrix_valid(up)||!lease.input||!lease.weight_arena||!lease.input_arena||
       !lease.workspace||!lease.handle||!lease.stream||!out.interleaved||!out.activation||
       out.rows!=kRows||out.tokens!=kTokens||out.interleaved_values<2*branch_values||
       out.activation_values<branch_values||lease.weight_bytes<weight_bytes||
       lease.input_bytes<input_bytes||lease.workspace_bytes<512)
        throw std::invalid_argument("Gemma compact K1792 fixed operand lease");
    if(reinterpret_cast<std::uintptr_t>(lease.weight_arena)%16||
       reinterpret_cast<std::uintptr_t>(lease.input_arena)%16||
       reinterpret_cast<std::uintptr_t>(out.interleaved)%16||
       reinterpret_cast<std::uintptr_t>(lease.workspace)%256)
        throw std::invalid_argument("Gemma compact K1792 cuBLAS arena alignment");
    const void* pointers[]{lease.input,lease.weight_arena,lease.input_arena,
        lease.workspace,out.interleaved,out.activation};
    const size_t bytes[]{inputs*sizeof(float),weight_bytes,input_bytes,
        lease.workspace_bytes,2*branch_values*sizeof(float),branch_values*sizeof(float)};
    for(int first=0;first<6;++first)
        for(int second=first+1;second<6;++second)
            if(overlaps(pointers[first],bytes[first],pointers[second],bytes[second]))
                throw std::invalid_argument("Gemma compact K1792 arena alias");
    const primitive::Matrix matrices[]{gate,up};
    for(auto matrix:matrices){
        const size_t raw_bytes=(size_t(kRows)*kHidden/256)*(matrix.quant==primitive::Quant::Q4K?144:210);
        for(int index=1;index<6;++index)
            if(overlaps(matrix.data,raw_bytes,pointers[index],bytes[index]))
                throw std::invalid_argument("Gemma compact K1792 raw weight lifetime");
    }
    auto* scalar=static_cast<float*>(lease.workspace);
    auto* w=static_cast<half*>(lease.weight_arena);
    auto* x=static_cast<half*>(lease.input_arena);
    initialize_input_scalars<<<1,1,0,lease.stream>>>(scalar);
    input_maximum<<<unsigned((inputs+255)/256),256,0,lease.stream>>>(lease.input,inputs,scalar);
    choose_input_scale<<<1,1,0,lease.stream>>>(scalar);
    panel_weights<<<unsigned((kMergedRows*kHidden+255)/256),256,0,lease.stream>>>(gate,up,w);
    panel_input<<<unsigned((inputs+255)/256),256,0,lease.stream>>>(lease.input,x,scalar);
    primitive::check_cuda(cudaGetLastError(),"Gemma compact K1792 direct encoding");
    blas(cublasSetStream(lease.handle,lease.stream));
    blas(cublasSetWorkspace(lease.handle,static_cast<unsigned char*>(lease.workspace)+256,lease.workspace_bytes-256));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_DEVICE));
    blas(cublasSetMathMode(lease.handle,static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    for(int segment=0;segment<kSegments;++segment){
        const half* a=w+size_t(segment)*kMergedRows*kPanel;
        const half* b=x+size_t(segment)*kTokens*kPanel;
        blas(cublasGemmEx(lease.handle,CUBLAS_OP_T,CUBLAS_OP_N,int(kMergedRows),kTokens,kPanel,
            scalar+2,a,CUDA_R_16F,kPanel,b,CUDA_R_16F,kPanel,scalar+(segment?5:4),
            out.interleaved,CUDA_R_32F,int(kMergedRows),
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    blas(cublasSetMathMode(lease.handle,CUBLAS_DEFAULT_MATH));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_HOST));
}
}
