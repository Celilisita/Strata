#include "fp4_gu_fused.hpp"
#include <nvfp4_layout.cuh>
#include <math.cuh>
#include "strata/qwen35/native/ffn.hpp"
#include <cmath>

namespace strata::qwen35::native {
namespace {
constexpr int Features=64,Tokens=128,BPitch=20;
struct __align__(16) Stage {
    uint4 a[512];
    unsigned b[Tokens*BPitch],asf[256],bsf[256];
};
struct __align__(16) Shared {Stage stages[2];};
static_assert(sizeof(Shared)==40960,"two bounded20KiB stages");
__device__ __forceinline__ void async16(void *dst,const void *src){
    const unsigned shared=unsigned(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(shared),"l"(src):"memory");
}
__device__ __forceinline__ void async4(void *dst,const void *src){
    const unsigned shared=unsigned(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"::"r"(shared),"l"(src):"memory");
}
__device__ __forceinline__ void commit(){asm volatile("cp.async.commit_group;":::"memory");}
__device__ __forceinline__ void wait(){asm volatile("cp.async.wait_group 0;":::"memory");}
__device__ __forceinline__ void prefetch(Stage &p,Nvfp4View g,Nvfp4View u,Nvfp4Input x,int feature,int token,int first){
    const int tid=threadIdx.x;
    #pragma unroll
    for(int pass=0;pass<2;++pass){
        const int index=tid+pass*256,m=index/64,part=(index/32)%2,lane=index%32;
        const auto w=m<4?g:u;
        const size_t tile=size_t(feature/16+m%4)*(g.columns/64)+first/64+part;
        async16(p.a+index,static_cast<const uint4*>(w.values)+tile*32+lane);
        const int row=index/4,column=index%4;
        const auto *src=static_cast<const uint4*>(x.values)+size_t(token+row)*(g.columns/32)+first/32+column;
        async16(p.b+row*BPitch+column*4,src);
    }
    const int role=tid/128,row=(tid/2)%64,part=tid%2;
    const auto weight=role?u:g;
    async4(p.asf+tid,static_cast<const unsigned char*>(weight.scales)+nvfp4_detail::scale_index(feature+row,first/16+4*part,g.columns));
    async4(p.bsf+tid,static_cast<const unsigned char*>(x.scales)+nvfp4_detail::scale_index(token+tid/2,first/16+4*part,g.columns));
}
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
template<bool Half,bool Debug> __global__ __launch_bounds__(256,2) void fused_gu(Nvfp4View g,Nvfp4View u,Nvfp4Input x,
    Nvfp4Input out,float global,float *gd,float *ud){
    __shared__ Shared p;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,m=warp/2,n=warp%2;
    const int feature=int(blockIdx.y)*Features,token=int(blockIdx.x)*Tokens;
    float ga[8][4]{},ua[8][4]{};
    prefetch(p.stages[0],g,u,x,feature,token,0);commit();wait();__syncthreads();
    int stage=0;
    for(int first=0;first<g.columns;first+=128){
        const int next=first+128;
        if(next<g.columns){prefetch(p.stages[stage^1],g,u,x,feature,token,next);commit();}
        const auto &current=p.stages[stage];
        #pragma unroll
        for(int part=0;part<2;++part){
            const uint4 a=current.a[(m*2+part)*32+lane],b=current.a[((m+4)*2+part)*32+lane];
            unsigned sa=0,su=0;
            if(lane%4<2){const int row=m*16+lane/4+8*(lane%4);
                sa=current.asf[row*2+part];su=current.asf[128+row*2+part];}
            #pragma unroll
            for(int fragment=0;fragment<8;++fragment){
                const int t=n*64+fragment*8+lane/4,k=part*8+lane%4;
                const unsigned b0=current.b[t*BPitch+k],b1=current.b[t*BPitch+k+4];
                const unsigned sb=lane%4==0?current.bsf[t*2+part]:0;
                mma(ga[fragment],a,b0,b1,sa,sb);mma(ua[fragment],b,b0,b1,su,sb);
            }
        }
        wait();__syncthreads();stage^=1;
    }
    auto *codes=reinterpret_cast<unsigned char*>(p.stages[0].a);
    emit<Half,Debug>(ga,ua,codes,out,feature,token,g.rows,g.weight_scale*g.input_scale,u.weight_scale*u.input_scale,global,gd,ud);
    __syncthreads();store(codes,out,feature,token,g.rows);
}
template<bool Half> __global__ void codec(const void *g,const void *u,Nvfp4Input out,int columns,float global){
    __shared__ __align__(16) unsigned char codes[4096];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,m=warp/2,n=warp%2;
    const int feature=int(blockIdx.y)*Features,token=int(blockIdx.x)*Tokens;
    float ga[8][4]{},ua[8][4]{};
    #pragma unroll
    for(int fragment=0;fragment<8;++fragment){
        #pragma unroll
        for(int i=0;i<4;++i){
            const int t=token+n*64+fragment*8+(lane%4)*2+(i&1),r=feature+m*16+lane/4+(i/2)*8;
            const size_t at=size_t(t)*columns+r;
            if constexpr(Half){ga[fragment][i]=__half2float(static_cast<const __half*>(g)[at]);ua[fragment][i]=__half2float(static_cast<const __half*>(u)[at]);}
            else{ga[fragment][i]=static_cast<const float*>(g)[at];ua[fragment][i]=static_cast<const float*>(u)[at];}
        }
    }
    emit<Half,false>(ga,ua,codes,out,feature,token,columns,1,1,global,nullptr,nullptr);
    __syncthreads();store(codes,out,feature,token,columns);
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
template<bool Half,bool Debug>void launch(PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,float *gd,float *ud,int tokens,float global,cudaStream_t stream){
    operands(g.view,u.view,x,out,tokens,global);
    if constexpr(Debug)if(!gd||!ud)throw std::invalid_argument("fused GU debug outputs");
    fused_gu<Half,Debug><<<dim3(tokens/Tokens,g.view.rows/Features),256,0,stream>>>(g.view,u.view,x,out,global,gd,ud);
    check_cuda(cudaGetLastError(),"fused packed GU SwiGLU FP4 encode");
}
}
void fp4_gu_fused_configure(){
    check_cuda(cudaFuncSetAttribute(fused_gu<false,false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"GU float shared carveout");
    check_cuda(cudaFuncSetAttribute(fused_gu<true,false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"GU half shared carveout");
}
GuFusionResources fp4_gu_fused_resources(bool half){
    cudaFuncAttributes attr{};int blocks=0;
    if(half){check_cuda(cudaFuncGetAttributes(&attr,fused_gu<true,false>),"GU half resources");
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fused_gu<true,false>,256,0),"GU half occupancy");}
    else{check_cuda(cudaFuncGetAttributes(&attr,fused_gu<false,false>),"GU float resources");
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fused_gu<false,false>,256,0),"GU float occupancy");}
    return {attr.numRegs,blocks,attr.sharedSizeBytes,attr.localSizeBytes};
}
void fp4_gu_fused(PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,int t,bool half,float global,cudaStream_t s){
    if(half)launch<true,false>(g,u,x,out,nullptr,nullptr,t,global,s);else launch<false,false>(g,u,x,out,nullptr,nullptr,t,global,s);
}
void fp4_gu_fused_debug(PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,float *gd,float *ud,int t,bool half,float global,cudaStream_t s){
    if(half)launch<true,true>(g,u,x,out,gd,ud,t,global,s);else launch<false,true>(g,u,x,out,gd,ud,t,global,s);
}
void fp4_gu_codec_probe(const void *g,const void *u,Nvfp4Input out,int t,int k,bool half,float global,cudaStream_t s){
    if(!g||!u||!out.values||!out.scales||t<128||t%128||k<64||k%64||t>out.capacity_rows||k>out.capacity_columns||!(global>0)||!std::isfinite(global))
        throw std::invalid_argument("GU codec fixture operands");
    if(half)codec<true><<<dim3(t/Tokens,k/Features),256,0,s>>>(g,u,out,k,global);
    else codec<false><<<dim3(t/Tokens,k/Features),256,0,s>>>(g,u,out,k,global);
    check_cuda(cudaGetLastError(),"GU exactcodec fixture");
}
}
