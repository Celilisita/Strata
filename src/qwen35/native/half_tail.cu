#include "half_tail.hpp"
#define gu_merge_math half_tail_math_reused
#include <gu_merge_math.cuh>
#undef gu_merge_math
#include <algorithm>
#include <vector>

namespace strata::qwen35::native {
namespace {
using namespace half_tail_math_reused;
template<bool Half,bool Debug> __device__ __forceinline__ void emit_tail(float (&g)[8][4],float (&u)[8][4],
    unsigned char *codes,Nvfp4Input out,int feature,int token,int columns,int tokens,float ag,float au,float global,float *gd,float *ud){
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
            if constexpr(Debug)if(token+t<tokens){const size_t at=size_t(token+t)*columns+feature+m*16+j;
                gd[at]=gl;gd[at+8]=gh;ud[at]=ul;ud[at+8]=uh;}
            const float x=swiglu_value(gl,ul),y=swiglu_value(gh,uh);
            float maximum=fmaxf(fabsf(x),fabsf(y));unsigned bad=unsigned(!isfinite(x)||!isfinite(y));
            #pragma unroll
            for(int d=4;d<=16;d*=2){maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));bad|=__shfl_xor_sync(0xffffffff,bad,d);}
            const unsigned char sf=bad?127:__nv_cvt_float_to_fp8(maximum/(6.f*global),__NV_SATFINITE,__NV_E4M3);
            const float scale=nvfp4_detail::scale_float(sf)*global,inv=scale>0?1.f/scale:0.f;
            const unsigned bits=__nv_cvt_float2_to_fp4x2(make_float2(x*inv,y*inv),__NV_E2M1,cudaRoundNearest);
            const unsigned neighbor=__shfl_xor_sync(0xffffffff,bits,4);
            if(!(j&1)){const int at=t*64+m*8+j/2;
                codes[at]=static_cast<unsigned char>((bits&15)|((neighbor&15)<<4));
                codes[at+4]=static_cast<unsigned char>((bits>>4)|(neighbor&240));}
            if(!j)static_cast<unsigned char*>(out.scales)[nvfp4_detail::scale_index(token+t,(feature+m*16)/16,columns)]=token+t<tokens?sf:0;
        }
    }
}
__device__ __forceinline__ void store_tail(const unsigned char *codes,Nvfp4Input out,int feature,int token,int columns,int tokens){
    const int row=threadIdx.x/4,col=threadIdx.x%4;
    auto *dst=static_cast<uint4*>(out.values)+size_t(token+row)*(columns/32)+feature/32+col;
    if(token+row<tokens)*dst=reinterpret_cast<const uint4*>(codes)[threadIdx.x];
}

constexpr int Features=128,Tokens=128,StageBytes=27648;
struct __align__(512) Stage {
    uint4 a[1024];
    unsigned b[2048],asf[512],bsf[256];
};
struct __align__(512) Shared {Stage stages[2];unsigned long long barriers[2];};
static_assert(sizeof(Stage)==StageBytes&&sizeof(Shared)==55808,"two27KiB TMA stages plus barriers");
static_assert(sizeof(GuTmaMaps)==384&&alignof(GuTmaMaps)==64,"immutable by-value tensor descriptors");
__device__ __forceinline__ unsigned address(const void *p){return unsigned(__cvta_generic_to_shared(p));}
__device__ __forceinline__ void tensor_copy(const CUtensorMap *map,void *dst,int x,int y,unsigned barrier){
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 ::"r"(address(dst)),"l"(map),"r"(x),"r"(y),"r"(barrier):"memory");
}
__device__ __forceinline__ void linear_copy(void *dst,const void *src,unsigned barrier){
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 1024, [%2];"
                 ::"r"(address(dst)),"l"(src),"r"(barrier):"memory");
}
__device__ __forceinline__ void prefetch(Stage &p,unsigned long long *barrier,const GuTmaMaps &maps,
    Nvfp4View g,Nvfp4View u,Nvfp4Input x,int feature,int token,int first){
    if(threadIdx.x)return;
    const unsigned completion=address(barrier);
    asm volatile("{ .reg .b64 state; mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 state, [%0], %1; }"
                 ::"r"(completion),"r"(StageBytes):"memory");
    tensor_copy(&maps.gate,p.a,first/64*128,feature/16,completion);
    tensor_copy(&maps.up,p.a+512,first/64*128,feature/16,completion);
    tensor_copy(&maps.input,p.b,first/8,token,completion);
    const size_t offset=size_t(feature/128)*(g.columns/64)*512+size_t(first/64)*512;
    linear_copy(p.asf,static_cast<const unsigned char*>(g.scales)+offset,completion);
    linear_copy(p.asf+256,static_cast<const unsigned char*>(u.scales)+offset,completion);
    linear_copy(p.bsf,static_cast<const unsigned char*>(x.scales)+size_t(token/128)*(g.columns/64)*512+size_t(first/64)*512,completion);
}
__device__ __forceinline__ void wait(unsigned long long *barrier,unsigned phase){
    unsigned ready=0;
    do{asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
                   "selp.b32 %0, 1, 0, p; }":"=r"(ready):"r"(address(barrier)),"r"(phase):"memory");}while(!ready);
}
__device__ __forceinline__ int swizzle(int word){return word^(((word>>5)&3)<<2);}
template<bool Debug> __global__ __launch_bounds__(512,1) void half_tail_kernel(const __grid_constant__ GuTmaMaps maps,
    Nvfp4View g,Nvfp4View u,Nvfp4Input x,Nvfp4Input out,float global,float *gd,float *ud,int tokens){
    extern __shared__ __align__(512) unsigned char arena[];
    auto &p=*reinterpret_cast<Shared*>(arena);
    if(!threadIdx.x){
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(address(p.barriers)):"memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(address(p.barriers+1)):"memory");
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    __syncthreads();
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,m=warp/2,n=warp%2;
    const int feature=int(blockIdx.y)*Features,token=int(blockIdx.x)*Tokens;
    float ga[8][4]{},ua[8][4]{};
    prefetch(p.stages[0],p.barriers,maps,g,u,x,feature,token,0);
    int stage=0;
    for(int first=0;first<g.columns;first+=128){
        wait(p.barriers+stage,unsigned(first/256)&1);
        if(first+128<g.columns)prefetch(p.stages[stage^1],p.barriers+(stage^1),maps,g,u,x,feature,token,first+128);
        const auto &current=p.stages[stage];
        #pragma unroll
        for(int part=0;part<2;++part){
            const uint4 a=current.a[(m*2+part)*32+lane],b=current.a[((m+8)*2+part)*32+lane];
            unsigned sa=0,su=0;
            if(lane%4<2){const int row=feature%128+m*16+lane/4+8*(lane%4);
                const int word=part*128+(row%32)*4+row/32;
                sa=current.asf[word];su=current.asf[256+word];}
            #pragma unroll
            for(int fragment=0;fragment<8;++fragment){
                const int t=n*64+fragment*8+lane/4,k=part*8+lane%4;
                const unsigned b0=current.b[swizzle(t*16+k)],b1=current.b[swizzle(t*16+k+4)];
                const unsigned sb=lane%4==0?current.bsf[part*128+(t%32)*4+t/32]:0;
                mma(ga[fragment],a,b0,b1,sa,sb);mma(ua[fragment],b,b0,b1,su,sb);
            }
        }
        __syncthreads();stage^=1;
    }
    auto *codes=reinterpret_cast<unsigned char*>(p.stages[0].a);
    emit_tail<true,Debug>(ga,ua,codes,out,feature,token,g.rows,tokens,g.weight_scale*g.input_scale,u.weight_scale*u.input_scale,global,gd,ud);
    __syncthreads();store_tail(codes,out,feature,token,g.rows,tokens);
    if(!threadIdx.x){
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(address(p.barriers)):"memory");
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(address(p.barriers+1)):"memory");
    }
}
template<bool Debug>void launch(const GuMergePlan &plan,PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,
    float *gd,float *ud,int tokens,bool half,float global,cudaStream_t stream){
    if(tokens!=5424)throw std::invalid_argument("Half tail fixed5424 geometry");
    operands(g.view,u.view,x,out,5504,global);
    if(g.view.rows%Features)throw std::invalid_argument("mergedGU fullfeature tile");
    if(!half||plan.gate!=g.view.values||plan.up!=u.view.values||plan.input!=x.values||
       plan.columns!=g.view.columns||plan.rows!=g.view.rows||plan.capacity!=x.capacity_rows||plan.capacity!=8192||x.capacity_columns!=5120||
       out.capacity_rows!=8192||g.view.rows!=17408||g.view.columns!=5120)
        throw std::invalid_argument("GU TMA prepared descriptor lease / Half-only prototype");
    const auto span=[](const void*p,size_t n,size_t align){const auto a=reinterpret_cast<uintptr_t>(p);
        if(!a||a%align||n>UINTPTR_MAX-a)throw std::invalid_argument("Half tail span/alignment");return std::pair<uintptr_t,uintptr_t>{a,a+n};};
    const auto overlap=[](auto a,auto b){return a.first<b.second&&b.first<a.second;};
    std::vector<std::pair<uintptr_t,uintptr_t>> reads{span(x.values,nvfp4_value_bytes(5504,5120),16),span(x.scales,nvfp4_scale_bytes(5504,5120),4),
        span(g.view.values,nvfp4_value_bytes(17408,5120),16),span(u.view.values,nvfp4_value_bytes(17408,5120),16),
        span(g.view.scales,nvfp4_scale_bytes(17408,5120),4),span(u.view.scales,nvfp4_scale_bytes(17408,5120),4)};
    std::vector<std::pair<uintptr_t,uintptr_t>> writes{span(out.values,nvfp4_value_bytes(tokens,17408),16),span(out.scales,nvfp4_scale_bytes(5504,17408),4)};
    if constexpr(Debug){writes.push_back(span(gd,size_t(tokens)*17408*4,4));writes.push_back(span(ud,size_t(tokens)*17408*4,4));}
    for(size_t i=0;i<writes.size();++i){for(auto r:reads)if(overlap(writes[i],r))throw std::invalid_argument("Half tail live input/write alias");
        for(size_t j=0;j<i;++j)if(overlap(writes[i],writes[j]))throw std::invalid_argument("Half tail write/write alias");}
    half_tail_kernel<Debug><<<dim3((tokens+Tokens-1)/Tokens,g.view.rows/Features),512,sizeof(Shared),stream>>>(plan.maps,g.view,u.view,x,out,global,gd,ud,tokens);
    check_cuda(cudaGetLastError(),"GU TMA packed half Swiglu FP4");
}
}
void half_tail_configure(){
    check_cuda(cudaFuncSetAttribute(half_tail_kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"merged GU dynamicshared");
    check_cuda(cudaFuncSetAttribute(half_tail_kernel<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"merged GU debug dynamicshared");
    check_cuda(cudaFuncSetAttribute(half_tail_kernel<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"GU TMA carveout");
}
HalfTailResources half_tail_resources(){
    cudaFuncAttributes a{},b{};int x=0,y=0;check_cuda(cudaFuncGetAttributes(&a,half_tail_kernel<false>),"Half tail resources");
    check_cuda(cudaFuncGetAttributes(&b,half_tail_kernel<true>),"Half tail debug resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&x,half_tail_kernel<false>,512,sizeof(Shared)),"Half tail occupancy");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&y,half_tail_kernel<true>,512,sizeof(Shared)),"Half tail debug occupancy");
    return{std::max(a.numRegs,b.numRegs),std::min(x,y),std::min(a.binaryVersion,b.binaryVersion),2,sizeof(Shared),a.localSizeBytes+b.localSizeBytes};
}
void half_tail(const GuMergePlan &p,PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,int t,bool half,float global,cudaStream_t s){
    launch<false>(p,g,u,x,out,nullptr,nullptr,t,half,global,s);
}
void half_tail_debug(const GuMergePlan &p,PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,float *gd,float *ud,int t,bool half,float global,cudaStream_t s){
    launch<true>(p,g,u,x,out,gd,ud,t,half,global,s);
}
}
