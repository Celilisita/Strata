#include "swiglu_vector.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_layout.cuh"
#include "math.cuh"
#include <cmath>
namespace strata::qwen35::native {
namespace {
template<bool Diagnostic> __global__ void vector_swiglu(const float *gate,const float *up,float *activation,
    unsigned *q,unsigned char *sf,int tokens,int columns,float global){
    // Two adjacent lanes own a block16 scale. Each lane loads two float4s
    // and stores four packed pairs, deleting repeated address/reduction work.
    const size_t group8=size_t(blockIdx.x)*blockDim.x+threadIdx.x,element=group8*8;
    if(element>=size_t(tokens)*columns)return;
    float x[8];float maximum=0;unsigned bad=0;
    #pragma unroll
    for(int i=0;i<2;++i){
        const float4 g=reinterpret_cast<const float4*>(gate)[group8*2+i];
        const float4 u=reinterpret_cast<const float4*>(up)[group8*2+i];
        x[i*4]=swiglu_value(g.x,u.x);x[i*4+1]=swiglu_value(g.y,u.y);
        x[i*4+2]=swiglu_value(g.z,u.z);x[i*4+3]=swiglu_value(g.w,u.w);
        if constexpr(Diagnostic)reinterpret_cast<float4*>(activation)[group8*2+i]=
            make_float4(x[i*4],x[i*4+1],x[i*4+2],x[i*4+3]);
    }
    #pragma unroll
    for(int i=0;i<8;++i){maximum=fmaxf(maximum,fabsf(x[i]));bad|=unsigned(!isfinite(x[i]));}
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1,2));
    bad|=__shfl_xor_sync(0xffffffff,bad,1,2);
    const unsigned char scale=bad?127:__nv_cvt_float_to_fp8(maximum/(6.f*global),__NV_SATFINITE,__NV_E4M3);
    const float restored=nvfp4_detail::scale_float(scale)*global,inverse=restored>0?1.f/restored:0;
    const int row=int(element/columns),column=int(element%columns);
    if(!(threadIdx.x&1))sf[nvfp4_detail::scale_index(row,column/16,columns)]=scale;
    unsigned packed=0;
    #pragma unroll
    for(int i=0;i<4;++i)packed|=unsigned(__nv_cvt_float2_to_fp4x2(
        make_float2(x[i*2]*inverse,x[i*2+1]*inverse),__NV_E2M1,cudaRoundNearest))<<(i*8);
    q[group8]=packed;
}
}
void nvfp4_swiglu_vector(const float *g,const float *u,float *diagnostic,Nvfp4Input x,int t,int k,float scale,cudaStream_t stream){
    if(!g||!u||!x.values||!x.scales||t<1||k<64||k%64||t>x.capacity_rows||k>x.capacity_columns||reinterpret_cast<size_t>(g)%16||reinterpret_cast<size_t>(u)%16||
       reinterpret_cast<size_t>(x.values)%4||(diagnostic&&reinterpret_cast<size_t>(diagnostic)%16)||!(scale>0)||!std::isfinite(scale))
        throw std::invalid_argument("NVFP4 vector SwiGLU geometry");
    const unsigned blocks=unsigned((size_t(t)*k/8+255)/256);
    if(diagnostic)vector_swiglu<true><<<blocks,256,0,stream>>>(g,u,diagnostic,static_cast<unsigned*>(x.values),static_cast<unsigned char*>(x.scales),t,k,scale);
    else vector_swiglu<false><<<blocks,256,0,stream>>>(g,u,nullptr,static_cast<unsigned*>(x.values),static_cast<unsigned char*>(x.scales),t,k,scale);
    check_cuda(cudaGetLastError(),"NVFP4 vector SwiGLU launch");
}
}
