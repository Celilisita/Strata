#include "fp4_gu_merge.hpp"
#include "gu_merge_math.cuh"
#include "gu_tma_encoder.hpp"

namespace strata::qwen35::native {
namespace {
using namespace gu_merge_math;
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
template<bool Debug> __global__ __launch_bounds__(512,1) void fused_gu_merge(const __grid_constant__ GuTmaMaps maps,
    Nvfp4View g,Nvfp4View u,Nvfp4Input x,Nvfp4Input out,float global,float *gd,float *ud){
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
    emit<true,Debug>(ga,ua,codes,out,feature,token,g.rows,g.weight_scale*g.input_scale,u.weight_scale*u.input_scale,global,gd,ud);
    __syncthreads();store(codes,out,feature,token,g.rows);
    if(!threadIdx.x){
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(address(p.barriers)):"memory");
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(address(p.barriers+1)):"memory");
    }
}
CUtensorMap make_map(const void *data,cuuint64_t width,cuuint64_t height,cuuint32_t bx,cuuint32_t by,bool swizzled){
    alignas(64) CUtensorMap map{};const cuuint64_t dims[2]={width,height},strides[1]={width*4};
    const cuuint32_t box[2]={bx,by},steps[2]={1,1};
    const auto code=tma_detail::encoder()(&map,CU_TENSOR_MAP_DATA_TYPE_UINT32,2,const_cast<void*>(data),dims,strides,box,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,swizzled?CU_TENSOR_MAP_SWIZZLE_64B:CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(code!=CUDA_SUCCESS)throw std::runtime_error("GU tensor map encode "+std::to_string(int(code)));
    return map;
}
template<bool Debug>void launch(const GuMergePlan &plan,PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,
    float *gd,float *ud,int tokens,bool half,float global,cudaStream_t stream){
    operands(g.view,u.view,x,out,tokens,global);
    if(g.view.rows%Features)throw std::invalid_argument("mergedGU fullfeature tile");
    if(!half||plan.gate!=g.view.values||plan.up!=u.view.values||plan.input!=x.values||
       plan.columns!=g.view.columns||plan.rows!=g.view.rows||plan.capacity!=x.capacity_rows)
        throw std::invalid_argument("GU TMA prepared descriptor lease / Half-only prototype");
    if constexpr(Debug)if(!gd||!ud)throw std::invalid_argument("GU TMA debug outputs");
    fused_gu_merge<Debug><<<dim3(tokens/Tokens,g.view.rows/Features),512,sizeof(Shared),stream>>>(plan.maps,g.view,u.view,x,out,global,gd,ud);
    check_cuda(cudaGetLastError(),"GU TMA packed half Swiglu FP4");
}
}
GuMergePlan make_gu_merge_plan(PackedFp4 g,PackedFp4 u,Nvfp4Input x){
    if(g.view.rows!=u.view.rows||g.view.columns!=u.view.columns||g.view.columns!=5120||g.view.rows!=17408||
       x.capacity_rows!=8192||x.capacity_columns!=5120)throw std::invalid_argument("GU TMA measured descriptor geometry");
    const auto k=cuuint64_t(g.view.columns);
    return {{make_map(g.view.values,k/64*128,g.view.rows/16,256,8,false),
             make_map(u.view.values,k/64*128,u.view.rows/16,256,8,false),
             make_map(x.values,k/8,x.capacity_rows,16,128,true)},g.view.values,u.view.values,x.values,g.view.columns,g.view.rows,x.capacity_rows};
}
void fp4_gu_merge_configure(){
    check_cuda(cudaFuncSetAttribute(fused_gu_merge<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"merged GU dynamicshared");
    check_cuda(cudaFuncSetAttribute(fused_gu_merge<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"merged GU debug dynamicshared");
    check_cuda(cudaFuncSetAttribute(fused_gu_merge<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"GU TMA carveout");
}
GuFusionResources fp4_gu_merge_resources(bool half){
    if(!half)throw std::invalid_argument("GU TMA prototype Half-only");
    cudaFuncAttributes attr{};int blocks=0;check_cuda(cudaFuncGetAttributes(&attr,fused_gu_merge<false>),"GU TMA resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fused_gu_merge<false>,512,sizeof(Shared)),"GU TMA occupancy");
    return {attr.numRegs,blocks,attr.sharedSizeBytes+sizeof(Shared),attr.localSizeBytes};
}
void fp4_gu_merge(const GuMergePlan &p,PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,int t,bool half,float global,cudaStream_t s){
    launch<false>(p,g,u,x,out,nullptr,nullptr,t,half,global,s);
}
void fp4_gu_merge_debug(const GuMergePlan &p,PackedFp4 g,PackedFp4 u,Nvfp4Input x,Nvfp4Input out,float *gd,float *ud,int t,bool half,float global,cudaStream_t s){
    launch<true>(p,g,u,x,out,gd,ud,t,half,global,s);
}
}
