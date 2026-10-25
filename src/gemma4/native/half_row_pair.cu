#include "half_row_pair.hpp"
#include "codec.cuh"
#include "library_input.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <limits>

namespace strata::gemma4::native {
namespace {
bool overlaps(const void* a,size_t a_bytes,const void* b,size_t b_bytes){
    const auto first=reinterpret_cast<std::uintptr_t>(a),second=reinterpret_cast<std::uintptr_t>(b);
    return first<=second?second-first<a_bytes:first-second<b_bytes;
}
size_t branch_values(const HalfRowPairOutput& output){
    if(!output.interleaved||!output.activation||output.rows<1||
       output.rows>(std::numeric_limits<int>::max)()/2||output.tokens<128||output.tokens>8192)
        throw std::invalid_argument("Gemma row-pair output geometry");
    const size_t count=size_t(output.rows)*output.tokens;
    if(count>size_t((std::numeric_limits<int>::max)())/2||
       output.interleaved_values<2*count||output.activation_values<count||
       overlaps(output.interleaved,2*count*sizeof(float),output.activation,count*sizeof(float)))
        throw std::invalid_argument("Gemma row-pair interleaved output lease");
    return count;
}
bool matrix_valid(primitive::Matrix matrix){
    return matrix.data&&matrix.rows>0&&matrix.columns>=256&&matrix.columns%256==0&&
        matrix.layout==primitive::BlockLayout::Gguf&&
        (matrix.quant==primitive::Quant::Q4K||matrix.quant==primitive::Quant::Q6K);
}
void blas(cublasStatus_t status){
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma FP16 merged-row cuBLAS projection");
}
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
__global__ void gelu_interleaved(HalfRowPairOutput output,size_t count){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=count)return;
    const size_t at=(i/output.rows)*size_t(2*output.rows)+i%output.rows;
    const float x=output.interleaved[at],up=output.interleaved[at+output.rows];
    const float t=0.7978845608028654f*(x+0.044715f*x*x*x);
    output.activation[i]=(0.5f*x*(1.f+tanhf(t)))*up;
}
__global__ void unpack_interleaved(HalfRowPairOutput output,float* gate,float* up,size_t count){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=count)return;
    const size_t at=(i/output.rows)*size_t(2*output.rows)+i%output.rows;
    gate[i]=output.interleaved[at];up[i]=output.interleaved[at+output.rows];
}
}

void half_row_pair_project(primitive::Matrix gate,primitive::Matrix up,const HalfRowPairLease& lease){
    const size_t output_values=branch_values(lease.output);
    if(!matrix_valid(gate)||!matrix_valid(up)||gate.rows!=up.rows||gate.columns!=up.columns||
       lease.output.rows!=gate.rows||!lease.input||!lease.weight_arena||!lease.input_arena||
       !lease.workspace||!lease.handle||!lease.stream||lease.workspace_bytes<512)
        throw std::invalid_argument("Gemma merged-row operand lease");
    const size_t weights=size_t(gate.rows)*gate.columns;
    const size_t inputs=size_t(lease.output.tokens)*gate.columns;
    const size_t weight_bytes=2*weights*sizeof(half),input_bytes=inputs*sizeof(half);
    if(weights>size_t((std::numeric_limits<int>::max)())/2||inputs>size_t((std::numeric_limits<int>::max)())||
       lease.weight_bytes<weight_bytes||lease.input_bytes<input_bytes)
        throw std::invalid_argument("Gemma merged-row encoding capacity");
    const void* pointers[]{lease.input,lease.weight_arena,lease.input_arena,
        lease.workspace,lease.output.interleaved,lease.output.activation};
    const size_t bytes[]{inputs*sizeof(float),weight_bytes,input_bytes,
        lease.workspace_bytes,2*output_values*sizeof(float),output_values*sizeof(float)};
    for(int first=0;first<6;++first)
        for(int second=first+1;second<6;++second)
            if(overlaps(pointers[first],bytes[first],pointers[second],bytes[second]))
                throw std::invalid_argument("Gemma merged-row arena alias");
    const primitive::Matrix matrices[]{gate,up};
    for(auto matrix:matrices){
        const size_t raw_bytes=(weights/256)*(matrix.quant==primitive::Quant::Q4K?144:210);
        for(int index=1;index<6;++index)
            if(overlaps(matrix.data,raw_bytes,pointers[index],bytes[index]))
                throw std::invalid_argument("Gemma merged-row raw weight lifetime");
    }
    auto* scalar=static_cast<float*>(lease.workspace);
    auto* encoded_weight=static_cast<half*>(lease.weight_arena);
    auto* encoded_input=static_cast<half*>(lease.input_arena);
    initialize_input_scalars<<<1,1,0,lease.stream>>>(scalar);
    input_maximum<<<unsigned((inputs+255)/256),256,0,lease.stream>>>(lease.input,inputs,scalar);
    choose_input_scale<<<1,1,0,lease.stream>>>(scalar);
    encode_weights<<<unsigned((2*weights+255)/256),256,0,lease.stream>>>(gate,up,encoded_weight,weights);
    encode_input<<<unsigned((inputs+255)/256),256,0,lease.stream>>>(lease.input,encoded_input,inputs,scalar);
    primitive::check_cuda(cudaGetLastError(),"Gemma FP16 merged-row encoding");
    blas(cublasSetStream(lease.handle,lease.stream));
    blas(cublasSetWorkspace(lease.handle,static_cast<unsigned char*>(lease.workspace)+256,lease.workspace_bytes-256));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_DEVICE));
    blas(cublasSetMathMode(lease.handle,static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    for(int first=0;first<gate.columns;first+=896){
        const int count=gate.columns-first<896?gate.columns-first:896;
        blas(cublasGemmEx(lease.handle,CUBLAS_OP_T,CUBLAS_OP_N,2*gate.rows,lease.output.tokens,count,
            scalar+2,encoded_weight+first,CUDA_R_16F,gate.columns,
            encoded_input+first,CUDA_R_16F,gate.columns,scalar+(first?5:4),
            lease.output.interleaved,CUDA_R_32F,2*gate.rows,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    blas(cublasSetMathMode(lease.handle,CUBLAS_DEFAULT_MATH));
    blas(cublasSetPointerMode(lease.handle,CUBLAS_POINTER_MODE_HOST));
}

void half_row_pair_gelu(const HalfRowPairOutput& output,cudaStream_t stream){
    const size_t count=branch_values(output);
    if(!stream)throw std::invalid_argument("Gemma merged-row GELU stream");
    gelu_interleaved<<<unsigned((count+255)/256),256,0,stream>>>(output,count);
    primitive::check_cuda(cudaGetLastError(),"Gemma merged-row GELU product");
}

void half_row_pair_unpack(const HalfRowPairOutput& output,float* gate,float* up,
    size_t capacity,cudaStream_t stream){
    const size_t count=branch_values(output),bytes=count*sizeof(float);
    if(!gate||!up||!stream||capacity<count||overlaps(gate,bytes,up,bytes)||
       overlaps(gate,bytes,output.interleaved,2*bytes)||overlaps(up,bytes,output.interleaved,2*bytes)||
       overlaps(gate,bytes,output.activation,bytes)||overlaps(up,bytes,output.activation,bytes))
        throw std::invalid_argument("Gemma merged-row diagnostic unpack lease");
    unpack_interleaved<<<unsigned((count+255)/256),256,0,stream>>>(output,gate,up,count);
    primitive::check_cuda(cudaGetLastError(),"Gemma merged-row diagnostic unpack");
}
}
