#include "strata/gemma4/native/vector_single.hpp"
#include "strata/gemma4/native/q6_down.hpp"
#include "strata/gemma4/native/q6_aligned.hpp"
#include "small_prefill_values.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::gemma4::native {
namespace {
bool overlaps(const void* a,size_t a_bytes,const void* b,size_t b_bytes){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<a_bytes:x-y<b_bytes;
}

__global__ void encode_single(const float* input,SingleQ8Block* output,int columns){
    const int group=blockIdx.x*4+threadIdx.x/32;
    const int lane=threadIdx.x%32,token=blockIdx.y;
    if(group>=columns/32)return;
    const float value=input[size_t(token)*columns+group*32+lane];
    const unsigned invalid=__ballot_sync(0xffffffff,!isfinite(value));
    float maximum=fabsf(value);
    for(int distance=16;distance;distance/=2)
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,distance));
    const float scale=invalid?nanf(""):maximum/127.f;
    const int code=invalid||!(scale>0)?0:max(-127,min(127,__float2int_rn(value/scale)));
    auto& block=output[size_t(token)*(columns/32)+group];
    block.code[lane]=static_cast<signed char>(code);
    int sum=code;
    for(int distance=16;distance;distance/=2)
        sum+=__shfl_xor_sync(0xffffffff,sum,distance);
    if(!lane){block.d=scale;block.sum=sum;}
}

// Keep the admitted grouped layout: eight lanes per row, one lane per full
// 32-value weight group. Only the two residual planes are removed.
template<primitive::Quant Type>
__device__ float term(const unsigned char* raw,const SingleQ8Block& input,
    int group,float previous){
    int first=0,second=0;
    #pragma unroll
    for(int word=0;word<8;++word){
        const int weight=primitive::detail::weight_codes<Type>(raw,group,word*4);
        const int code=reinterpret_cast<const int*>(input.code)[word];
        if(word<4)first=__dp4a(weight,code,first);
        else second=__dp4a(weight,code,second);
    }
    if constexpr(Type==primitive::Quant::Q4K){
        const unsigned metadata=primitive::detail::metadata(raw,group,false);
        const float scale=__half2float(*reinterpret_cast<const half*>(raw))*float(metadata&63);
        const float minimum=__half2float(*reinterpret_cast<const half*>(raw+2))*float(metadata>>6);
        const float product=__fmaf_rn(input.d,float(first+second),0.f);
        const float correction=__fmaf_rn(input.d,float(input.sum),0.f);
        return __fadd_rn(previous,__fmaf_rn(scale,product,-__fmul_rn(minimum,correction)));
    }else{
        // Q6K has two signed scales per 32 values, one for each 16-value half.
        // weight_codes retains its signed [-32,31] packing from the parent.
        const auto* scales=reinterpret_cast<const signed char*>(raw+192)+group*2;
        const int dot=int(scales[0])*first+int(scales[1])*second;
        const float product=__fmaf_rn(input.d,float(dot),0.f);
        return __fmaf_rn(__half2float(*reinterpret_cast<const half*>(raw+208)),product,previous);
    }
}

__device__ float sum8(float value,unsigned active){
    for(int distance=4;distance;distance/=2)
        value=__fadd_rn(value,__shfl_xor_sync(active,value,distance));
    return value;
}

template<primitive::Quant A,primitive::Quant B,int Tokens,bool Pair>
__global__ void multiply_single(primitive::Matrix first,primitive::Matrix second,
    const SingleQ8Block* input,float* gate,float* up,float* activation){
    const int row=blockIdx.x*16+threadIdx.x/8,group=threadIdx.x%8;
    if(row>=first.rows)return;
    const unsigned active=__activemask();
    constexpr int BytesA=A==primitive::Quant::Q4K?144:210;
    constexpr int BytesB=B==primitive::Quant::Q4K?144:210;
    const int blocks=first.columns/256;
    float a[Tokens]{},b[Tokens]{};
    for(int k=0;k<blocks;++k){
        const auto* p=static_cast<const unsigned char*>(first.data)+(size_t(row)*blocks+k)*BytesA;
        const auto* q=Pair?static_cast<const unsigned char*>(second.data)+(size_t(row)*blocks+k)*BytesB:nullptr;
        #pragma unroll
        for(int token=0;token<Tokens;++token){
            const auto& code=input[size_t(token)*(first.columns/32)+k*8+group];
            a[token]=term<A>(p,code,group,a[token]);
            if constexpr(Pair)b[token]=term<B>(q,code,group,b[token]);
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
       matrix.layout!=primitive::BlockLayout::Gguf||
       (matrix.quant!=primitive::Quant::Q4K&&matrix.quant!=primitive::Quant::Q6K)||
       !input.data||input.columns!=matrix.columns||input.tokens<1||input.tokens>3||
       input.bytes<size_t(input.columns/32)*input.tokens*sizeof(SingleQ8Block)||
       reinterpret_cast<std::uintptr_t>(input.data)%alignof(SingleQ8Block)||!stream)
        throw std::invalid_argument("Gemma single-Q8 typed operand lease");
}

template<primitive::Quant A,primitive::Quant B,bool Pair>
void launch(primitive::Matrix first,primitive::Matrix second,SingleQ8View input,
    float* gate,float* up,float* activation,cudaStream_t stream){
    const int grid=(first.rows+15)/16;
    if(input.tokens==1)multiply_single<A,B,1,Pair><<<grid,128,0,stream>>>(first,second,input.data,gate,up,activation);
    else if(input.tokens==2)multiply_single<A,B,2,Pair><<<grid,128,0,stream>>>(first,second,input.data,gate,up,activation);
    else multiply_single<A,B,3,Pair><<<grid,128,0,stream>>>(first,second,input.data,gate,up,activation);
}

template<bool Pair>
void dispatch(primitive::Matrix first,primitive::Matrix second,SingleQ8View input,
    float* gate,float* up,float* activation,cudaStream_t stream){
    using Q=primitive::Quant;
    if(first.quant==Q::Q4K){
        if(second.quant==Q::Q4K)launch<Q::Q4K,Q::Q4K,Pair>(first,second,input,gate,up,activation,stream);
        else launch<Q::Q4K,Q::Q6K,Pair>(first,second,input,gate,up,activation,stream);
    }else{
        if(second.quant==Q::Q4K)launch<Q::Q6K,Q::Q4K,Pair>(first,second,input,gate,up,activation,stream);
        else launch<Q::Q6K,Q::Q6K,Pair>(first,second,input,gate,up,activation,stream);
    }
    primitive::check_cuda(cudaGetLastError(),"Gemma independent single-Q8 grouped projection");
}
}

SingleQ8View prepare_single_q8(const float* input,void* arena,size_t arena_bytes,
    int columns,int tokens,cudaStream_t stream){
    if(!input||!arena||columns<256||columns%256||tokens<1||tokens>3||!stream||
       reinterpret_cast<std::uintptr_t>(arena)%alignof(SingleQ8Block))
        throw std::invalid_argument("Gemma single-Q8 activation extent/alignment");
    const size_t bytes=size_t(columns/32)*tokens*sizeof(SingleQ8Block);
    if(bytes>arena_bytes||overlaps(input,size_t(columns)*tokens*sizeof(float),arena,bytes))
        throw std::invalid_argument("Gemma single-Q8 activation arena lifetime");
    auto* output=static_cast<SingleQ8Block*>(arena);
    encode_single<<<dim3((columns/32+3)/4,tokens),128,0,stream>>>(input,output,columns);
    primitive::check_cuda(cudaGetLastError(),"Gemma single-Q8 GPU encoding");
    return {output,bytes,columns,tokens};
}

void linear_single_q8(primitive::Matrix matrix,SingleQ8View input,float* output,cudaStream_t stream){
    if(is_aligned_q6_down(matrix)){
        const size_t bytes=q6_down_weight_bytes(matrix);
        linear_aligned_q6({matrix.data,bytes,matrix.rows,matrix.columns},input,output,stream);
        return;
    }
    validate(matrix,input,stream);
    if(!output||overlaps(input.data,input.bytes,output,size_t(matrix.rows)*input.tokens*sizeof(float)))
        throw std::invalid_argument("Gemma single-Q8 linear output lifetime");
    dispatch<false>(matrix,matrix,input,output,nullptr,nullptr,stream);
}

void pair_single_q8(primitive::Matrix first,primitive::Matrix second,SingleQ8View input,
    float* gate,float* up,float* activation,cudaStream_t stream){
    validate(first,input,stream);validate(second,input,stream);
    if(!gate||!up||!activation||first.rows!=second.rows||first.columns!=second.columns)
        throw std::invalid_argument("Gemma single-Q8 pair geometry");
    const size_t bytes=size_t(first.rows)*input.tokens*sizeof(float);
    if(overlaps(gate,bytes,up,bytes)||overlaps(gate,bytes,activation,bytes)||overlaps(up,bytes,activation,bytes)||
       overlaps(input.data,input.bytes,gate,bytes)||overlaps(input.data,input.bytes,up,bytes)||
       overlaps(input.data,input.bytes,activation,bytes))
        throw std::invalid_argument("Gemma single-Q8 pair output lifetime");
    dispatch<true>(first,second,input,gate,up,activation,stream);
}
}
