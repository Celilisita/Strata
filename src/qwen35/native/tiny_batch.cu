#include "strata/qwen35/native/tiny_batch.hpp"
#include "vector_fragment.cuh"
#include "math.cuh"
namespace strata::qwen35::native {
namespace {
using detail::ActivationBlock;
__global__ void encode(const float *x,ActivationBlock *out,int columns){
    const int group=int(blockIdx.x)*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(group>=columns/32)return;
    const float value=x[size_t(blockIdx.y)*columns+group*32+lane];float maximum=fabsf(value);
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    const float delta=__fmul_rn(maximum,1.0f/127.0f);
    const int code=maximum?int(roundf(__fdividef(value,delta))):0;
    auto &b=out[size_t(blockIdx.y)*(columns/32)+group];b.codes[lane]=static_cast<signed char>(code);
    if(!lane)b.delta=__float2half_rn(delta);
}
template<Quant Type,bool Pair,int Tokens,int Parts,bool Packed>
__global__ void multiply(Matrix a,Matrix b,const ActivationBlock *input,float *first,float *second,float *activation){
    constexpr int Logical=128,Extra=Parts==2?1:0,Workers=Type==Quant::Q4K?16:32;
    constexpr int Bytes=Type==Quant::Q4K?144:210;
    __shared__ float partial[Tokens][Pair?2:1][Logical];
    const int row=int(blockIdx.x),tid=threadIdx.x,lane=tid%32;
    float one[Tokens][Extra+1]{},two[Tokens][Extra+1]{};
    #pragma unroll
    for(int stripe=0;stripe<=Extra;++stripe){
        const int logical=tid+stripe*64;
        for(int block=logical/Workers;block<a.columns/256;block+=Logical/Workers){
            const size_t offset=(size_t(row)*(a.columns/256)+block)*Bytes;
            const auto *ap=static_cast<const unsigned char *>(a.data)+offset;
            const auto *bp=static_cast<const unsigned char *>(b.data)+offset;
            #pragma unroll
            for(int t=0;t<Tokens;++t){
                const auto *x=input+size_t(t)*(a.columns/32)+block*8;
                one[t][stripe]=detail::accumulate_fragment<Type,Packed>(ap,x,one[t][stripe],nullptr);
                if constexpr(Pair)two[t][stripe]=detail::accumulate_fragment<Type,Packed>(bp,x,two[t][stripe],nullptr);
            }
        }
        #pragma unroll
        for(int t=0;t<Tokens;++t){partial[t][0][logical]=one[t][stripe];if constexpr(Pair)partial[t][1][logical]=two[t][stripe];}
    }
    __syncthreads();
    if(tid<32){
        #pragma unroll
        for(int t=0;t<Tokens;++t){
            float x=partial[t][0][lane],y=0;if constexpr(Pair)y=partial[t][1][lane];
            #pragma unroll
            for(int stripe=1;stripe<4;++stripe){x=__fadd_rn(x,partial[t][0][stripe*32+lane]);if constexpr(Pair)y=__fadd_rn(y,partial[t][1][stripe*32+lane]);}
            for(int d=16;d;d/=2){x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));if constexpr(Pair)y=__fadd_rn(y,__shfl_xor_sync(0xffffffff,y,d));}
            if(!lane){const size_t at=size_t(t)*a.rows+row;first[at]=x;if constexpr(Pair){second[at]=y;activation[at]=swiglu_value(x,y);}}
        }
    }
}
template<bool Pair,int Tokens> void launch(Matrix a,Matrix b,const void *input,float *out,float *second,float *act,cudaStream_t s){
    if(a.quant==Quant::Q4K){
        constexpr int Parts=Pair?4:2;
        if(a.layout==BlockLayout::Q4Pairs)multiply<Quant::Q4K,Pair,Tokens,Parts,true><<<a.rows,Parts*32,0,s>>>(a,b,static_cast<const ActivationBlock *>(input),out,second,act);
        else multiply<Quant::Q4K,Pair,Tokens,Parts,false><<<a.rows,Parts*32,0,s>>>(a,b,static_cast<const ActivationBlock *>(input),out,second,act);
    }else multiply<Quant::Q6K,Pair,Tokens,4,false><<<a.rows,128,0,s>>>(a,b,static_cast<const ActivationBlock *>(input),out,second,act);
}
void validate(Matrix m,const void *x,const float *out,int tokens){
    matrix_bytes(m);
    if(!x||!out||(tokens!=2&&tokens!=3)||(m.layout!=BlockLayout::Gguf&&m.layout!=BlockLayout::Q4Pairs))
        throw std::invalid_argument("invalid own tiny-batch matrix lease");
}
}
void tiny_batch_quantize(const float *x,void *out,int columns,int tokens,cudaStream_t s){
    if(!x||!out||columns<256||columns%256||(tokens!=2&&tokens!=3))throw std::invalid_argument("invalid tiny activation span");
    encode<<<dim3((columns/32+3)/4,tokens),128,0,s>>>(x,static_cast<ActivationBlock *>(out),columns);
    check_cuda(cudaGetLastError(),"own tiny batch quantization");
}
void tiny_batch_linear(Matrix m,const void *x,float *out,int tokens,cudaStream_t s){
    validate(m,x,out,tokens);
    if(tokens==2)launch<false,2>(m,m,x,out,nullptr,nullptr,s);else launch<false,3>(m,m,x,out,nullptr,nullptr,s);
    check_cuda(cudaGetLastError(),"own tiny batch projection");
}
void tiny_batch_pair(Matrix a,Matrix b,const void *x,float *first,float *second,float *act,int tokens,cudaStream_t s){
    validate(a,x,first,tokens);validate(b,x,second,tokens);
    if(!act||a.quant!=b.quant||a.layout!=b.layout||a.rows!=b.rows||a.columns!=b.columns)throw std::invalid_argument("invalid tiny pair");
    if(tokens==2)launch<true,2>(a,b,x,first,second,act,s);else launch<true,3>(a,b,x,first,second,act,s);
    check_cuda(cudaGetLastError(),"own tiny batch fused pair");
}
}
