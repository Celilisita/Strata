#include "nvfp4_q8_fastsoa.cuh"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "nvfp4_codec.cuh"
#include "nvfp4_q8_fast_codec.cuh"
#include "math.cuh"
#include <math_constants.h>
#include <cuda_fp16.h>
namespace strata::completed_v100::qwen35::native {
namespace {
struct alignas(16) Input {int4 code;};
static_assert(sizeof(Input)==16,"Threecodeplanes plusoneFloat4scaleplane use64bytes/group");
__global__ void encode(const float* input,Input* output,int groups) {
    const int group=(blockIdx.x*blockDim.x+threadIdx.x)/16,lane=threadIdx.x&15;
    if(group>=groups)return;
    const float value=input[size_t(blockIdx.y)*groups*16+group*16+lane/2+(lane&1)*8];
    float maximum=fabsf(value);
    for(int d=8;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,16));
    const bool invalid=__any_sync(0xffffffff,!isfinite(value));
    const float scale=maximum>0.f&&maximum/127.f>0.f?maximum/127.f:1.f;
    const int q=__float2int_rn(value/scale);unsigned code=unsigned(q)&255;
    #pragma unroll
    for(int j=1;j<4;++j)code|=(unsigned(__shfl_down_sync(0xffffffff,q,j,16))&255)<<(8*j);
    auto* target=output+size_t(blockIdx.y)*groups+group;
    auto* scales=reinterpret_cast<float4*>(output+3*groups)+group;
    if(!(lane&3))reinterpret_cast<int*>(&target->code)[lane/4]=int(code);
    if(!lane){reinterpret_cast<float*>(scales)[blockIdx.y]=invalid?CUDART_NAN_F:scale;if(!blockIdx.y)scales->w=0.f;}
}
__global__ void reconstruct(const Input* input,float* output,int columns,int tokens) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(columns)*tokens) {
        const int groups=columns/16,g=int((i%columns)/16),token=int(i/columns);
        const auto& group=input[token*groups+g];
        const int code=reinterpret_cast<const signed char*>(&group.code)[2*(i%8)+(i%16)/8];
        output[i]=float(code)*reinterpret_cast<const float*>(reinterpret_cast<const float4*>(input+3*groups)+g)[token];
    }
}
struct Weight {int4 code;float delta;};
__device__ __forceinline__ Weight read_weight(const unsigned char* block,int group) {
    const auto* p=reinterpret_cast<const unsigned*>(block+4+(group&3)*8);
    const unsigned lo=__ldg(p),hi=__ldg(p+1);
    return {make_int4(q8fast::unpack_fast(lo,false),q8fast::unpack_fast(lo,true),q8fast::unpack_fast(hi,false),q8fast::unpack_fast(hi,true)),
        nvfp4::block_scale(__ldg(block+(group&3)))};
}
__device__ __forceinline__ int dot(int4 w,int4 x) {
    int value=__dp4a(w.x,x.x,0);value=__dp4a(w.y,x.y,value);
    value=__dp4a(w.z,x.z,value);return __dp4a(w.w,x.w,value);
}
__device__ __forceinline__ float4 scales4(const float4* p) {
    float4 s;
    asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(s.x),"=f"(s.y),"=f"(s.z),"=f"(s.w) : "l"(p));
    return s;
}
template<int Tokens,bool Pair> __global__ void project_soa(const unsigned char* gate,const float* gscale,
    const unsigned char* up,const float* uscale,const Input* input,float* first,float* second,float* activation,int rows,int columns) {
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;if(row>=rows)return;
    const int groups=columns/16;const auto* a=gate+size_t(row)*((columns/64)*36);
    const auto* b=Pair?up+size_t(row)*(columns/64)*36:nullptr;float one[Tokens]{},two[Tokens]{};
    for(int group=lane;group<groups;group+=32) {
        const auto wg=read_weight(a+(group/4)*36,group);
        Weight wu{};if constexpr(Pair)wu=read_weight(b+(group/4)*36,group);
        const float4 delta=scales4(reinterpret_cast<const float4*>(input+3*groups)+group);
        #pragma unroll
        for(int token=0;token<Tokens;++token) {
            const int4 code=__ldg(&(input+size_t(token)*groups+group)->code);const float scale=token==0?delta.x:token==1?delta.y:delta.z;
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
            const size_t i=size_t(t)*rows+row;const float x=one[t]*gscale[0];first[i]=x;
            if constexpr(Pair){const float y=two[t]*uscale[0];second[i]=y;activation[i]=swiglu_value(x,y);}
        }
    }
}
template<bool Pair> void launch(const void* a,const float* sa,const void* b,const float* sb,const void* input,
    float* out,float* second,float* act,int rows,int columns,int tokens,cudaStream_t s) {
    const dim3 grid(unsigned((rows+3)/4));
    if(tokens==1)project_soa<1,Pair><<<grid,128,0,s>>>(static_cast<const unsigned char*>(a),sa,static_cast<const unsigned char*>(b),sb,static_cast<const Input*>(input),out,second,act,rows,columns);
    else if(tokens==2)project_soa<2,Pair><<<grid,128,0,s>>>(static_cast<const unsigned char*>(a),sa,static_cast<const unsigned char*>(b),sb,static_cast<const Input*>(input),out,second,act,rows,columns);
    else project_soa<3,Pair><<<grid,128,0,s>>>(static_cast<const unsigned char*>(a),sa,static_cast<const unsigned char*>(b),sb,static_cast<const Input*>(input),out,second,act,rows,columns);
}
}
void nvfp4_q8fs_prepare(const float* x,void* y,int columns,int tokens,cudaStream_t s) {
    if(!x||!y||columns<64||columns%64||tokens!=3)throw std::invalid_argument("Q8activation precision probe extent");
    encode<<<dim3(unsigned((columns+127)/128),unsigned(tokens)),128,0,s>>>(x,static_cast<Input*>(y),columns/16);
    check_cuda(cudaGetLastError(),"Explicit Q8activation encode");
}
void nvfp4_q8fs_reconstruct(const void* x,float* y,int columns,int tokens,cudaStream_t s) {
    if(!x||!y||static_cast<const void*>(y)==x||columns<64||columns%64||tokens!=3)
        throw std::invalid_argument("Q8activation reconstructed input extent");
    reconstruct<<<unsigned((size_t(columns)*tokens+255)/256),256,0,s>>>(static_cast<const Input*>(x),y,columns,tokens);
    check_cuda(cudaGetLastError(),"Q8activation precision oracle input");
}
void nvfp4_q8fs_project(const void* a,const float* sa,const void* b,const float* sb,const void* x,
    float* first,float* second,float* activation,int rows,int columns,int tokens,cudaStream_t s) {
    if(!a||!sa||!x||!first||rows<1||columns<64||columns%64||tokens!=3||bool(b)!=bool(sb)||bool(b)!=bool(second)||bool(b)!=bool(activation)||static_cast<const void*>(first)==x||
       (b&&(first==second||first==activation||second==activation||static_cast<const void*>(second)==x||static_cast<const void*>(activation)==x)))
        throw std::invalid_argument("Q8activation original weight owner/pair extent");
    if(b)launch<true>(a,sa,b,sb,x,first,second,activation,rows,columns,tokens,s);
    else launch<false>(a,sa,nullptr,nullptr,x,first,nullptr,nullptr,rows,columns,tokens,s);
    check_cuda(cudaGetLastError(),"Q8activation original NVFP4 integer projection");
}
}
