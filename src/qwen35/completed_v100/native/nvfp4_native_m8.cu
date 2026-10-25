#include "nvfp4_native_m8.cuh"
#include "original_fused_codec.cuh"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "math.cuh"
#include <math_constants.h>
namespace strata::completed_v100::qwen35::native {
namespace {
__global__ void encode(const float* input,half* output,int columns,int tokens) {
    const int group=(blockIdx.x*blockDim.x+threadIdx.x)/16,lane=threadIdx.x&15;
    if(group>=columns/16)return;
    const int column=group*16+original_fused::nvfp4_k(lane);
    const float value=input[size_t(blockIdx.y)*columns+column];float maximum=fabsf(value);
    for(int d=8;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,16));
    const unsigned invalid=__ballot_sync(0xffffffff,!isfinite(value));
    const bool bad=(invalid&(0xffffu<<(threadIdx.x&16)))!=0;
    const float delta=maximum>0.f&&maximum/127.f>0.f?maximum/127.f:1.f;
    const float quantized=bad?CUDART_NAN_F:float(__float2int_rn(value/delta))*delta;
    const half high=__float2half_rn(quantized);
    const size_t i=size_t(blockIdx.y)*columns+group*16+lane;
    output[i]=high;output[size_t(tokens)*columns+i]=__float2half_rn((quantized-__half2float(high))*4096.f);
}
__device__ __forceinline__ void mma(float* c,uint2 a,uint2 b) {
#if __CUDA_ARCH__ >= 700
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
        : "r"(a.x),"r"(a.y),"r"(b.x),"r"(b.y));
#else
    asm volatile("trap;"); // Unsupported capability must fail, never return a zero product.
#endif
}
template<bool Pair> __global__ void project(const unsigned char* gate,const unsigned char* up,const half* input,
    float* partial,int rows,int columns,int tokens,int chunk,int splits) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32,quad=(lane>>2)&3;
    const int fragment_row=(lane&3)+(lane>=16?4:0);
    const int tile=(blockIdx.x*4+warp)*32+quad*8;
    const int weight_row=tile+fragment_row;
    const auto* raw=(Pair&&blockIdx.z?up:gate);
    const int begin=blockIdx.y*chunk,end=min(begin+chunk,columns);
    float accum[8]{};
    for(int k=begin;k<end;k+=16) {
        __align__(16) half weights[16];
        if(weight_row<rows) {
            const auto* block=raw+(size_t(weight_row)*(columns/64)+k/64)*36;
            const unsigned scale=__ldg(block+(k&63)/16);
            const auto* codes=reinterpret_cast<const unsigned*>(block+4+((k&63)/16)*8);
            original_fused::nvfp4_eight_values(scale,__ldg(codes),weights);
            original_fused::nvfp4_eight_values(scale,__ldg(codes+1),weights+8);
        }else {
            #pragma unroll
            for(int i=0;i<16;++i)weights[i]=__float2half_rn(0.f);
        }
        #pragma unroll
        for(int step=0;step<4;++step) {
            const uint2 a=fragment_row<2*tokens?__ldg(reinterpret_cast<const uint2*>(input+size_t(fragment_row)*columns+k+4*step)):make_uint2(0,0);
            const uint2 b=*reinterpret_cast<const uint2*>(weights+4*step);mma(accum,a,b);
        }
    }
    #pragma unroll
    for(int i=0;i<8;++i) {
        const int m=(lane&1)+(i&2)+(lane>=16?4:0),n=tile+(i&4)+(lane&2)+(i&1);
        if(m<2*tokens&&n<rows)partial[((size_t(blockIdx.z)*splits+blockIdx.y)*2*tokens+m)*rows+n]=accum[i];
    }
}
template<bool Pair> __global__ void reduce(const float* partial,const float* scale,const float* up_scale,
    float* first,float* second,float* activation,int rows,int tokens,int splits) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(rows)*tokens)return;
    const int token=int(i/rows),row=int(i%rows);float high=0.f,low=0.f,uh=0.f,ul=0.f;
    for(int split=0;split<splits;++split) {
        const size_t at=(size_t(split)*2*tokens+token)*rows+row;
        high+=partial[at];low+=partial[at+size_t(tokens)*rows];
        if constexpr(Pair){const size_t offset=size_t(splits)*2*tokens*rows;uh+=partial[offset+at];ul+=partial[offset+at+size_t(tokens)*rows];}
    }
    const float g=fmaf(low,1.f/4096.f,high)*scale[0];first[i]=g;
    if constexpr(Pair){const float u=fmaf(ul,1.f/4096.f,uh)*up_scale[0];second[i]=u;activation[i]=swiglu_value(g,u);}
}
template<bool Pair> void launch(const void* g,const float* sg,const void* u,const float* su,const void* encoded,
    float* first,float* second,float* activation,int rows,int columns,int tokens,int splits,float* partial,cudaStream_t stream) {
    const int chunk=((columns+splits-1)/splits+63)/64*64;
    const dim3 grid(unsigned((rows+127)/128),unsigned(splits),Pair?2:1);
    project<Pair><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(g),static_cast<const unsigned char*>(u),
        static_cast<const half*>(encoded),partial,rows,columns,tokens,chunk,splits);
    check_cuda(cudaGetLastError(),"Native M8 quantized partial");
    reduce<Pair><<<unsigned((size_t(rows)*tokens+255)/256),256,0,stream>>>(partial,sg,su,first,second,activation,rows,tokens,splits);
    check_cuda(cudaGetLastError(),"Native M8 complete reduction/activation");
}
}
void nvfp4_m8_prepare(const float* input,void* encoded,int columns,int tokens,cudaStream_t stream) {
    if(!input||!encoded||static_cast<const void*>(input)==encoded||columns<64||columns%64||tokens<1||tokens>3)
        throw std::invalid_argument("Native M8 quantized activation extent/alias");
    encode<<<dim3(unsigned((columns+127)/128),unsigned(tokens)),128,0,stream>>>(input,static_cast<half*>(encoded),columns,tokens);
    check_cuda(cudaGetLastError(),"Native M8 Q8 high/low activation preparation");
}
void nvfp4_m8_project(const void* g,const float* sg,const void* u,const float* su,const void* encoded,
    float* first,float* second,float* activation,int rows,int columns,int tokens,int splits,
    float* partial,size_t capacity,cudaStream_t stream) {
    if(!g||!sg||!encoded||!first||!partial||rows<1||columns<64||columns%64||tokens<1||tokens>3||splits<1||splits>32||
       bool(u)!=bool(su)||bool(u)!=bool(second)||bool(u)!=bool(activation)||
       capacity<size_t(u?2:1)*splits*2*tokens*rows||static_cast<const void*>(first)==encoded||static_cast<const void*>(partial)==encoded||first==partial||
       (u&&(first==second||first==activation||second==activation||second==partial||activation==partial||static_cast<const void*>(second)==encoded||static_cast<const void*>(activation)==encoded)))
        throw std::invalid_argument("Native M8 original NVFP4/pair/partial ownership");
    if(u)launch<true>(g,sg,u,su,encoded,first,second,activation,rows,columns,tokens,splits,partial,stream);
    else launch<false>(g,sg,nullptr,nullptr,encoded,first,nullptr,nullptr,rows,columns,tokens,splits,partial,stream);
}
}
