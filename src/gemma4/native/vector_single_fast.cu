#include "strata/gemma4/native/vector_single_fast.hpp"
#include "small_prefill_values.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::gemma4::native {
namespace {
bool overlaps(const void* a,size_t a_bytes,const void* b,size_t b_bytes){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<a_bytes:x-y<b_bytes;
}
__device__ unsigned aligned_metadata(const unsigned char* raw,int group){
    const unsigned shift=unsigned(group&3)*8;
    const unsigned scale_byte=(*reinterpret_cast<const unsigned*>(raw+4)>>shift)&255u;
    const unsigned minimum_byte=(*reinterpret_cast<const unsigned*>(raw+8)>>shift)&255u;
    if(group<4)return (scale_byte&63u)|((minimum_byte&63u)<<6);
    const unsigned high=(*reinterpret_cast<const unsigned*>(raw+12)>>shift)&255u;
    const unsigned scale=(high&15u)|((scale_byte>>6)<<4);
    const unsigned minimum=(high>>4)|((minimum_byte>>6)<<4);
    return scale|(minimum<<6);
}
__device__ float term(const unsigned char* raw,const SingleQ8Block& input,
    int group,float previous){
    int first=0,second=0;
    #pragma unroll
    for(int word=0;word<8;++word){
        const int weight=primitive::detail::weight_codes<primitive::Quant::Q4K,true>(raw,group,word*4);
        const int code=reinterpret_cast<const int*>(input.code)[word];
        if(word<4)first=__dp4a(weight,code,first);
        else second=__dp4a(weight,code,second);
    }
    const unsigned metadata=aligned_metadata(raw,group);
    const unsigned delta=*reinterpret_cast<const unsigned*>(raw);
    const float scale=__half2float(__ushort_as_half(static_cast<unsigned short>(delta)))*float(metadata&63);
    const float minimum=__half2float(__ushort_as_half(static_cast<unsigned short>(delta>>16)))*float(metadata>>6);
    const float product=__fmaf_rn(input.d,float(first+second),0.f);
    const float correction=__fmaf_rn(input.d,float(input.sum),0.f);
    return __fadd_rn(previous,__fmaf_rn(scale,product,-__fmul_rn(minimum,correction)));
}
__device__ float sum8(float value,unsigned active){
    for(int distance=4;distance;distance/=2)
        value=__fadd_rn(value,__shfl_xor_sync(active,value,distance));
    return value;
}
template<int Tokens,bool Pair>
__global__ void multiply(primitive::Matrix first,primitive::Matrix second,
    const SingleQ8Block* input,float* gate,float* up,float* activation){
    const int row=blockIdx.x*16+threadIdx.x/8,group=threadIdx.x%8;
    if(row>=first.rows)return;
    const unsigned active=__activemask();
    const int blocks=first.columns/256;
    float a[Tokens]{},b[Tokens]{};
    for(int k=0;k<blocks;++k){
        const auto* p=static_cast<const unsigned char*>(first.data)+(size_t(row)*blocks+k)*144;
        const auto* q=Pair?static_cast<const unsigned char*>(second.data)+(size_t(row)*blocks+k)*144:nullptr;
        #pragma unroll
        for(int token=0;token<Tokens;++token){
            const auto& code=input[size_t(token)*(first.columns/32)+k*8+group];
            a[token]=term(p,code,group,a[token]);
            if constexpr(Pair)b[token]=term(q,code,group,b[token]);
        }
    }
    #pragma unroll
    for(int token=0;token<Tokens;++token){
        const float value=sum8(a[token],active);
        float other=0;
        if constexpr(Pair)other=sum8(b[token],active);
        if(!group){
            const size_t at=size_t(token)*first.rows+row;gate[at]=value;
            if constexpr(Pair){
                up[at]=other;
                const float z=0.7978845608028654f*(value+0.044715f*value*value*value);
                activation[at]=(0.5f*value*(1.f+tanhf(z)))*other;
            }
        }
    }
}
void validate(primitive::Matrix matrix,SingleQ8View input,cudaStream_t stream){
    if(!matrix.data||matrix.rows<1||matrix.columns<256||matrix.columns%256||
       matrix.layout!=primitive::BlockLayout::Gguf||matrix.quant!=primitive::Quant::Q4K||
       reinterpret_cast<std::uintptr_t>(matrix.data)%4||!input.data||
       input.columns!=matrix.columns||input.tokens<1||input.tokens>3||
       input.bytes<size_t(input.columns/32)*input.tokens*sizeof(SingleQ8Block)||
       reinterpret_cast<std::uintptr_t>(input.data)%alignof(SingleQ8Block)||!stream)
        throw std::invalid_argument("Gemma aligned Q4 single-Q8 typed operand lease");
}
template<bool Pair>
void launch(primitive::Matrix first,primitive::Matrix second,SingleQ8View input,
    float* gate,float* up,float* activation,cudaStream_t stream){
    const int grid=(first.rows+15)/16;
    if(input.tokens==1)multiply<1,Pair><<<grid,128,0,stream>>>(first,second,input.data,gate,up,activation);
    else if(input.tokens==2)multiply<2,Pair><<<grid,128,0,stream>>>(first,second,input.data,gate,up,activation);
    else multiply<3,Pair><<<grid,128,0,stream>>>(first,second,input.data,gate,up,activation);
    primitive::check_cuda(cudaGetLastError(),"Gemma aligned Q4 single-Q8 projection");
}
}

void linear_single_q8_fast(primitive::Matrix matrix,SingleQ8View input,
    float* output,cudaStream_t stream){
    if(matrix.quant!=primitive::Quant::Q4K){linear_single_q8(matrix,input,output,stream);return;}
    validate(matrix,input,stream);
    const size_t bytes=size_t(matrix.rows)*input.tokens*sizeof(float);
    const size_t raw_bytes=size_t(matrix.rows)*(matrix.columns/256)*144;
    if(!output||overlaps(input.data,input.bytes,output,bytes)||
       overlaps(matrix.data,raw_bytes,output,bytes))
        throw std::invalid_argument("Gemma aligned Q4 linear output lifetime");
    launch<false>(matrix,matrix,input,output,nullptr,nullptr,stream);
}
void pair_single_q8_fast(primitive::Matrix first,primitive::Matrix second,SingleQ8View input,
    float* gate,float* up,float* activation,cudaStream_t stream){
    if(first.quant!=primitive::Quant::Q4K||second.quant!=primitive::Quant::Q4K){
        pair_single_q8(first,second,input,gate,up,activation,stream);return;
    }
    validate(first,input,stream);validate(second,input,stream);
    if(!gate||!up||!activation||first.rows!=second.rows||first.columns!=second.columns)
        throw std::invalid_argument("Gemma aligned Q4 pair geometry");
    const size_t bytes=size_t(first.rows)*input.tokens*sizeof(float);
    if(overlaps(gate,bytes,up,bytes)||overlaps(gate,bytes,activation,bytes)||overlaps(up,bytes,activation,bytes)||
       overlaps(input.data,input.bytes,gate,bytes)||overlaps(input.data,input.bytes,up,bytes)||
       overlaps(input.data,input.bytes,activation,bytes))
        throw std::invalid_argument("Gemma aligned Q4 pair output lifetime");
    const size_t raw_bytes=size_t(first.rows)*(first.columns/256)*144;
    const void* outputs[]{gate,up,activation};
    for(const void* output:outputs)
        if(overlaps(first.data,raw_bytes,output,bytes)||overlaps(second.data,raw_bytes,output,bytes))
            throw std::invalid_argument("Gemma aligned Q4 raw weight lifetime");
    launch<true>(first,second,input,gate,up,activation,stream);
}
}
