#include "strata/gemma4/native/q6_aligned.hpp"
#include "small_prefill_values.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace strata::gemma4::native {
namespace {
constexpr size_t RawBytes=210,Stride=224;
bool aligned(const void* pointer,size_t alignment){
    return pointer&&reinterpret_cast<std::uintptr_t>(pointer)%alignment==0;
}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
size_t blocks_for(int rows,int columns){
    if(rows<1||rows>262144||columns<256||columns>21504||columns%256)
        throw std::invalid_argument("Gemma aligned Q6 matrix geometry");
    return size_t(rows)*(columns/256);
}
// Repack is cold and deliberately a byte copy. Its cost is reported separately
// from repeated resident projections, not hidden inside an inference timer.
__global__ void pack(const unsigned char* source,unsigned char* destination,size_t bytes){
    const size_t at=size_t(blockIdx.x)*256+threadIdx.x;
    if(at>=bytes)return;
    const size_t block=at/Stride,within=at%Stride;
    destination[at]=within<RawBytes?source[block*RawBytes+within]:0;
}
__device__ float term(const unsigned char* raw,const SingleQ8Block& input,int group,float prior){
    int first=0,second=0;
    #pragma unroll
    for(int word=0;word<8;++word){
        const int weight=primitive::detail::weight_codes<primitive::Quant::Q6K,true>(raw,group,word*4);
        const int code=reinterpret_cast<const int*>(input.code)[word];
        if(word<4)first=__dp4a(weight,code,first);
        else second=__dp4a(weight,code,second);
    }
    // Identical signed scales, integer product and F32 FMA order to196 Q6K.
    const auto* scales=reinterpret_cast<const signed char*>(raw+192)+group*2;
    const int dot=int(scales[0])*first+int(scales[1])*second;
    const float product=__fmaf_rn(input.d,float(dot),0.f);
    return __fmaf_rn(__half2float(*reinterpret_cast<const half*>(raw+208)),product,prior);
}
__device__ float sum8(float value,unsigned active){
    for(int distance=4;distance;distance/=2)value=__fadd_rn(value,__shfl_xor_sync(active,value,distance));
    return value;
}
template<int Tokens>
__global__ void multiply(AlignedQ6View matrix,const SingleQ8Block* input,float* output){
    const int row=blockIdx.x*16+threadIdx.x/8,group=threadIdx.x%8;
    if(row>=matrix.rows)return;
    const unsigned active=__activemask();const int blocks=matrix.columns/256;
    float accumulator[Tokens]{};
    for(int k=0;k<blocks;++k){
        const auto* raw=static_cast<const unsigned char*>(matrix.data)+(size_t(row)*blocks+k)*Stride;
        #pragma unroll
        for(int token=0;token<Tokens;++token){
            const auto& code=input[size_t(token)*(matrix.columns/32)+k*8+group];
            accumulator[token]=term(raw,code,group,accumulator[token]);
        }
    }
    #pragma unroll
    for(int token=0;token<Tokens;++token){
        const float value=sum8(accumulator[token],active);
        if(!group)output[size_t(token)*matrix.rows+row]=value;
    }
}
}
size_t aligned_q6_bytes(primitive::Matrix matrix){
    if(!matrix.data||matrix.quant!=primitive::Quant::Q6K||matrix.layout!=primitive::BlockLayout::Gguf)
        throw std::invalid_argument("Gemma aligned Q6 requires raw GGUF Q6K");
    return blocks_for(matrix.rows,matrix.columns)*Stride;
}
AlignedQ6View pack_aligned_q6(primitive::Matrix matrix,void* destination,size_t capacity,cudaStream_t stream){
    const size_t bytes=aligned_q6_bytes(matrix),raw_bytes=blocks_for(matrix.rows,matrix.columns)*RawBytes;
    if(!aligned(matrix.data,2)||!aligned(destination,32)||capacity<bytes||!stream||
       overlap(matrix.data,raw_bytes,destination,bytes))
        throw std::invalid_argument("Gemma aligned Q6 cold packing lifetime/capacity");
    pack<<<unsigned((bytes+255)/256),256,0,stream>>>(static_cast<const unsigned char*>(matrix.data),
        static_cast<unsigned char*>(destination),bytes);
    primitive::check_cuda(cudaGetLastError(),"Gemma exact Q6K 210-to224 cold packing");
    return {destination,bytes,matrix.rows,matrix.columns};
}
void linear_aligned_q6(AlignedQ6View matrix,SingleQ8View input,float* output,cudaStream_t stream){
    const size_t required=blocks_for(matrix.rows,matrix.columns)*Stride;
    if(!aligned(matrix.data,32)||matrix.bytes<required||!aligned(input.data,4)||
       input.tokens<1||input.tokens>3||input.columns!=matrix.columns||
       input.bytes<size_t(input.columns/32)*input.tokens*sizeof(SingleQ8Block)||!aligned(output,4)||!stream)
        throw std::invalid_argument("Gemma aligned Q6 typed input/output lease");
    const size_t result=size_t(matrix.rows)*input.tokens*sizeof(float);
    if(overlap(matrix.data,matrix.bytes,output,result)||overlap(input.data,input.bytes,output,result))
        throw std::invalid_argument("Gemma aligned Q6 projection output alias");
    const unsigned grid=(unsigned(matrix.rows)+15)/16;
    if(input.tokens==1)multiply<1><<<grid,128,0,stream>>>(matrix,input.data,output);
    else if(input.tokens==2)multiply<2><<<grid,128,0,stream>>>(matrix,input.data,output);
    else multiply<3><<<grid,128,0,stream>>>(matrix,input.data,output);
    primitive::check_cuda(cudaGetLastError(),"Gemma resident aligned Q6 single-Q8 projection");
}
}
