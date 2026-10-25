#pragma once
#include "fp4_gu_fused.hpp"
#include <nvfp4_layout.cuh>
#include <math.cuh>
#include "strata/qwen35/native/ffn.hpp"
#include <cmath>
namespace strata::qwen35::native::gu_tma_math {
__device__ __forceinline__ void mma(float (&c)[4],uint4 a,unsigned b0,unsigned b1,unsigned sa,unsigned sb){
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        :"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
}
template<bool Half,bool Debug> __device__ __forceinline__ void emit(float (&g)[8][4],float (&u)[8][4],
    unsigned char *codes,Nvfp4Input out,int feature,int token,int columns,float ag,float au,float global,float *gd,float *ud){
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,m=warp/2,n=warp%2,j=lane/4,local=lane%4;
    #pragma unroll
    for(int fragment=0;fragment<8;++fragment){
        #pragma unroll
        for(int parity=0;parity<2;++parity){
            float gl=__fmul_rn(g[fragment][parity],ag),gh=__fmul_rn(g[fragment][parity+2],ag);
            float ul=__fmul_rn(u[fragment][parity],au),uh=__fmul_rn(u[fragment][parity+2],au);
            if constexpr(Half){gl=__half2float(__float2half_rn(gl));gh=__half2float(__float2half_rn(gh));
                ul=__half2float(__float2half_rn(ul));uh=__half2float(__float2half_rn(uh));}
            const int t=n*64+fragment*8+local*2+parity;
            if constexpr(Debug){const size_t at=size_t(token+t)*columns+feature+m*16+j;
                gd[at]=gl;gd[at+8]=gh;ud[at]=ul;ud[at+8]=uh;}
            const float x=swiglu_value(gl,ul),y=swiglu_value(gh,uh);
            float maximum=fmaxf(fabsf(x),fabsf(y));unsigned bad=unsigned(!isfinite(x)||!isfinite(y));
            #pragma unroll
            for(int d=4;d<=16;d*=2){maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));bad|=__shfl_xor_sync(0xffffffff,bad,d);}
            const unsigned char sf=bad?127:__nv_cvt_float_to_fp8(maximum/(6.f*global),__NV_SATFINITE,__NV_E4M3);
            const float scale=nvfp4_detail::scale_float(sf)*global,inv=scale>0?1.f/scale:0.f;
            const unsigned bits=__nv_cvt_float2_to_fp4x2(make_float2(x*inv,y*inv),__NV_E2M1,cudaRoundNearest);
            const unsigned neighbor=__shfl_xor_sync(0xffffffff,bits,4);
            if(!(j&1)){const int at=t*32+m*8+j/2;
                codes[at]=static_cast<unsigned char>((bits&15)|((neighbor&15)<<4));
                codes[at+4]=static_cast<unsigned char>((bits>>4)|(neighbor&240));}
            if(!j)static_cast<unsigned char*>(out.scales)[nvfp4_detail::scale_index(token+t,(feature+m*16)/16,columns)]=sf;
        }
    }
}
__device__ __forceinline__ void store(const unsigned char *codes,Nvfp4Input out,int feature,int token,int columns){
    const int row=threadIdx.x/2,col=threadIdx.x%2;
    auto *dst=static_cast<uint4*>(out.values)+size_t(token+row)*(columns/32)+feature/32+col;
    *dst=reinterpret_cast<const uint4*>(codes)[threadIdx.x];
}
void operands(Nvfp4View g,Nvfp4View u,Nvfp4Input x,Nvfp4Input out,int tokens,float global){
    if(!g.values||!u.values||!g.scales||!u.scales||g.rows<64||g.rows%64||g.columns<128||g.columns%128||
       g.rows!=u.rows||g.columns!=u.columns||g.input_scale!=u.input_scale||tokens<128||tokens%128||
       !x.values||!x.scales||!out.values||!out.scales||tokens>x.capacity_rows||tokens>out.capacity_rows||
       g.columns>x.capacity_columns||g.rows>out.capacity_columns||!(global>0)||!std::isfinite(global)||
       !(g.weight_scale>0)||!(u.weight_scale>0)||!std::isfinite(g.weight_scale*g.input_scale)||!std::isfinite(u.weight_scale*u.input_scale))
        throw std::invalid_argument("fused GU packed operands/geometry/scale");
    auto overlap=[](const void *a,size_t n,const void *b,size_t m){
        const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+m&&y<x+n;
    };
    const size_t iv=nvfp4_value_bytes(tokens,g.columns),is=nvfp4_scale_bytes(tokens,g.columns);
    const size_t ov=nvfp4_value_bytes(tokens,g.rows),os=nvfp4_scale_bytes(tokens,g.rows);
    if(overlap(x.values,iv,out.values,ov)||overlap(x.values,iv,out.scales,os)||
       overlap(x.scales,is,out.values,ov)||overlap(x.scales,is,out.scales,os)||
       reinterpret_cast<uintptr_t>(x.values)%16||reinterpret_cast<uintptr_t>(g.values)%16||reinterpret_cast<uintptr_t>(u.values)%16||
       reinterpret_cast<uintptr_t>(x.scales)%4||reinterpret_cast<uintptr_t>(g.scales)%4||reinterpret_cast<uintptr_t>(u.scales)%4||
       reinterpret_cast<uintptr_t>(out.values)%16)
        throw std::invalid_argument("GU prepared operands must have distinct aligned lifetimes");
}
}
