#include "nvfp4_q8_activation.cuh"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_codec.cuh"
#include "math.cuh"
#include <math_constants.h>
#include <cuda_fp16.h>
namespace strata::qwen35::native {
namespace {
struct alignas(16) Input {int4 code;float scale;unsigned invalid,padding[2];};
static_assert(sizeof(Input)==32,"Q8activation group16 uses2bytes/value workspace");
__global__ void encode(const float* input,Input* output,int groups) {
    const int group=(blockIdx.x*blockDim.x+threadIdx.x)/16,lane=threadIdx.x&15;
    if(group>=groups)return;
    const float value=input[size_t(blockIdx.y)*groups*16+group*16+lane];
    float maximum=fabsf(value);
    for(int d=8;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,16));
    const bool invalid=__any_sync(0xffffffff,!isfinite(value));
    const float scale=maximum>0.f&&maximum/127.f>0.f?maximum/127.f:1.f;
    const int q=__float2int_rn(value/scale);unsigned code=unsigned(q)&255;
    #pragma unroll
    for(int j=1;j<4;++j)code|=(unsigned(__shfl_down_sync(0xffffffff,q,j,16))&255)<<(8*j);
    auto* target=output+size_t(blockIdx.y)*groups+group;
    if(!(lane&3))reinterpret_cast<int*>(&target->code)[lane/4]=int(code);
    if(!lane){target->scale=invalid?CUDART_NAN_F:scale;target->invalid=invalid;}
}
__global__ void reconstruct(const Input* input,float* output,int columns,int tokens) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(columns)*tokens) {
        const auto& group=input[(i/columns)*(columns/16)+(i%columns)/16];
        const int code=reinterpret_cast<const signed char*>(&group.code)[i%16];
        output[i]=float(code)*group.scale;
    }
}
__device__ __forceinline__ int unpack(unsigned word,bool high) {
    const unsigned code=(word>>(high?4:0))&0x0f0f0f0f,mag=code&0x07070707;
    const unsigned selector=(mag&7)|((mag>>4)&0x70)|((mag>>8)&0x700)|((mag>>12)&0x7000);
    const unsigned value=__byte_perm(0x03020100,0x0c080604,selector),sign=((code&0x08080808)>>3)*255u;
    return int(__vsub4(value^sign,sign));
}
struct Weight {int4 code;float delta;};
__device__ __forceinline__ Weight read_weight(const unsigned char* block,int group) {
    const auto* p=reinterpret_cast<const unsigned*>(block+4+(group&3)*8);
    const unsigned lo=__ldg(p),hi=__ldg(p+1);
    return {make_int4(unpack(lo,false),unpack(hi,false),unpack(lo,true),unpack(hi,true)),
        nvfp4::block_scale(__ldg(block+(group&3)))};
}
__device__ __forceinline__ int dot(int4 w,int4 x) {
    int value=__dp4a(w.x,x.x,0);value=__dp4a(w.y,x.y,value);
    value=__dp4a(w.z,x.z,value);return __dp4a(w.w,x.w,value);
}
__device__ __forceinline__ int word(const unsigned short* p) {
    return int(unsigned(__ldg(p))|(unsigned(__ldg(p+1))<<16));
}
__device__ __forceinline__ Weight read_q8(const unsigned char* row,int group) {
    const auto* block=row+(group/2)*34;
    const auto* p=reinterpret_cast<const unsigned short*>(block+2+(group&1)*16);
    return {make_int4(word(p),word(p+2),word(p+4),word(p+6)),__half2float(__ldg(reinterpret_cast<const half*>(block)))};
}
template<int Tokens,bool Pair,bool Q8=false> __global__ void project(const unsigned char* gate,const float* gscale,
    const unsigned char* up,const float* uscale,const Input* input,float* first,float* second,float* activation,int rows,int columns) {
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;if(row>=rows)return;
    const int groups=columns/16;const auto* a=gate+size_t(row)*(Q8?(columns/32)*34:(columns/64)*36);
    const auto* b=Pair?up+size_t(row)*(columns/64)*36:nullptr;float one[Tokens]{},two[Tokens]{};
    for(int group=lane;group<groups;group+=32) {
        const auto wg=Q8?read_q8(a,group):read_weight(a+(group/4)*36,group);
        Weight wu{};if constexpr(Pair)wu=read_weight(b+(group/4)*36,group);
        #pragma unroll
        for(int token=0;token<Tokens;++token) {
            const auto* x=input+size_t(token)*groups+group;const int4 code=__ldg(&x->code);
            const float scale=__ldg(&x->scale);
            one[token]=fmaf(float(dot(wg.code,code))*scale,wg.delta,one[token]);
            if constexpr(Pair)two[token]=fmaf(float(dot(wu.code,code))*scale,wu.delta,two[token]);
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        #pragma unroll
        for(int d=16;d;d/=2) {
            one[t]+=__shfl_down_sync(0xffffffff,one[t],d);
            if constexpr(Pair)two[t]+=__shfl_down_sync(0xffffffff,two[t],d);
        }
        if(!lane) {
            const size_t i=size_t(t)*rows+row;const float x=one[t]*(Q8?1.f:gscale[0]);first[i]=x;
            if constexpr(Pair){const float y=two[t]*uscale[0];second[i]=y;activation[i]=swiglu_value(x,y);}
        }
    }
}
template<bool Pair,bool Q8=false> void launch(const void* a,const float* sa,const void* b,const float* sb,const void* input,
    float* out,float* second,float* act,int rows,int columns,int tokens,cudaStream_t s) {
    const dim3 grid(unsigned((rows+3)/4));
    if(tokens==1)project<1,Pair,Q8><<<grid,128,0,s>>>(static_cast<const unsigned char*>(a),sa,static_cast<const unsigned char*>(b),sb,static_cast<const Input*>(input),out,second,act,rows,columns);
    else if(tokens==2)project<2,Pair,Q8><<<grid,128,0,s>>>(static_cast<const unsigned char*>(a),sa,static_cast<const unsigned char*>(b),sb,static_cast<const Input*>(input),out,second,act,rows,columns);
    else project<3,Pair,Q8><<<grid,128,0,s>>>(static_cast<const unsigned char*>(a),sa,static_cast<const unsigned char*>(b),sb,static_cast<const Input*>(input),out,second,act,rows,columns);
}
}
void nvfp4_q8a_prepare(const float* x,void* y,int columns,int tokens,cudaStream_t s) {
    if(!x||!y||columns<64||columns%64||tokens<1||tokens>3)throw std::invalid_argument("Q8activation precision probe extent");
    encode<<<dim3(unsigned((columns+127)/128),unsigned(tokens)),128,0,s>>>(x,static_cast<Input*>(y),columns/16);
    check_cuda(cudaGetLastError(),"Explicit Q8activation encode");
}
void nvfp4_q8a_reconstruct(const void* x,float* y,int columns,int tokens,cudaStream_t s) {
    if(!x||!y||static_cast<const void*>(y)==x||columns<64||columns%64||tokens<1||tokens>3)
        throw std::invalid_argument("Q8activation reconstructed input extent");
    reconstruct<<<unsigned((size_t(columns)*tokens+255)/256),256,0,s>>>(static_cast<const Input*>(x),y,columns,tokens);
    check_cuda(cudaGetLastError(),"Q8activation precision oracle input");
}
void nvfp4_q8a_project(const void* a,const float* sa,const void* b,const float* sb,const void* x,
    float* first,float* second,float* activation,int rows,int columns,int tokens,cudaStream_t s) {
    if(!a||!sa||!x||!first||rows<1||columns<64||columns%64||tokens<1||tokens>3||bool(b)!=bool(sb)||bool(b)!=bool(second)||bool(b)!=bool(activation)||static_cast<const void*>(first)==x||
       (b&&(first==second||first==activation||second==activation||static_cast<const void*>(second)==x||static_cast<const void*>(activation)==x)))
        throw std::invalid_argument("Q8activation original weight owner/pair extent");
    if(b)launch<true>(a,sa,b,sb,x,first,second,activation,rows,columns,tokens,s);
    else launch<false>(a,sa,nullptr,nullptr,x,first,nullptr,nullptr,rows,columns,tokens,s);
    check_cuda(cudaGetLastError(),"Q8activation original NVFP4 integer projection");
}
void raw_q8a_project(const void* raw,const void* input,float* output,int rows,int columns,int tokens,cudaStream_t s) {
    if(!raw||!input||!output||rows<1||columns<64||columns%64||tokens<1||tokens>3||static_cast<const void*>(output)==input)
        throw std::invalid_argument("Q8activation original Q8 projection extent");
    launch<false,true>(raw,nullptr,nullptr,nullptr,input,output,nullptr,nullptr,rows,columns,tokens,s);
    check_cuda(cudaGetLastError(),"Single Q8activation Q8weight projection");
}

}
