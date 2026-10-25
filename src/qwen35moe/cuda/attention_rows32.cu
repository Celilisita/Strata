#include "attention_rows32.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
namespace wm=nvcuda::wmma;
constexpr int Keys=64,Dim=256,Heads=16,CacheStride=512;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
cudaDeviceProp actual_sm75(){int device=0;cudaDeviceProp p{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&p,device));
    return p;}
__device__ float maximum(float x){for(int d=16;d;d>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,d));return x;}
__device__ float total(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffffu,x,d);return x;}
__global__ void zero_tail(half* k,half* v,int width,int cells){const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<cells){k[size_t(width)*CacheStride+i]=__float2half_rn(0);v[size_t(width)*CacheStride+i]=__float2half_rn(0);}}
template<int Rows> constexpr size_t shared_bytes(){return Rows*Dim*sizeof(half)+Rows*Keys*sizeof(float)+Rows*Keys*sizeof(half)+Rows*sizeof(float);}
static_assert(shared_bytes<32>()==28800);
template<int Rows,int Threads>
__global__ __launch_bounds__(Threads,1) void query_tile(const float* q,const half* k,const half* v,int first,int count,float* out){
#if __CUDA_ARCH__ == 750
    constexpr int Groups=2,Warps=Threads/32,Own=Rows/Warps;
    static_assert(Rows==(Threads/128)*32 && Own==8);
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,head=blockIdx.x,kv=head/8;
    const int column_warp=warp%4,row_group=(warp/4)*Groups,qbase=blockIdx.y*Rows;
    const int active=min(Rows,count-qbase),width=first+qbase+active;
    extern __shared__ __align__(32) unsigned char storage[];
    auto* query=reinterpret_cast<half*>(storage);
    auto* scores=reinterpret_cast<float*>(query+Rows*Dim);
    auto* probability=reinterpret_cast<half*>(scores+Rows*Keys);
    auto* row_scales=reinterpret_cast<float*>(probability+Rows*Keys);
    using Acc=wm::fragment<wm::accumulator,16,16,16,float>;
    Acc running[Groups][4];
#pragma unroll
    for(int g=0;g<Groups;++g){
#pragma unroll
        for(int j=0;j<4;++j)wm::fill_fragment(running[g][j],0.0f);
    }
    float previous_max[Own],previous_sum[Own];
#pragma unroll
    for(int own=0;own<Own;++own){previous_max[own]=-CUDART_INF_F;previous_sum[own]=0;}
    for(int i=tid;i<Rows*Dim;i+=Threads){const int row=i/Dim,d=i%Dim;
        const float value=row<active?q[(size_t(qbase+row)*Heads+head)*Dim+d]:0;
        query[i]=__hmul(__float2half_rn(value*0.0625f),__float2half_rn(0.25f));}
    __syncthreads();
    for(int key=0;key<width;key+=Keys){
        {
            Acc c[Groups];
#pragma unroll
            for(int g=0;g<Groups;++g)wm::fill_fragment(c[g],0.0f);
            // A single B load serves two distinct 16-query row fragments in this head.
            for(int d=0;d<Dim;d+=16){
                wm::fragment<wm::matrix_b,16,16,16,half,wm::col_major>b;
                wm::load_matrix_sync(b,k+size_t(key+column_warp*16)*CacheStride+kv*Dim+d,CacheStride);
#pragma unroll
                for(int g=0;g<Groups;++g){wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major>a;
                    wm::load_matrix_sync(a,query+(row_group+g)*16*Dim+d,Dim);wm::mma_sync(c[g],a,b,c[g]);}
            }
#pragma unroll
            for(int g=0;g<Groups;++g)wm::store_matrix_sync(scores+(row_group+g)*16*Keys+column_warp*16,c[g],Keys,wm::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for(int own=0;own<Own;++own){const int row=warp+own*Warps,end=first+qbase+row+1;const bool valid=row<active;
            const float s0=valid&&key+lane<end?scores[row*Keys+lane]*4.0f:-CUDART_INF_F;
            const float s1=valid&&key+lane+32<end?scores[row*Keys+lane+32]*4.0f:-CUDART_INF_F;
            const float m=valid?fmaxf(previous_max[own],maximum(fmaxf(s0,s1))):0;
            const float p0=valid?expf(s0-m):0,p1=valid?expf(s1-m):0;
            const float scale=valid?expf(previous_max[own]-m):0,sum=total(p0+p1);
            const float denominator=valid?previous_sum[own]*scale+sum:1;
            probability[row*Keys+lane]=__float2half_rn(p0);probability[row*Keys+lane+32]=__float2half_rn(p1);
            previous_max[own]=__shfl_sync(0xffffffffu,m,0);previous_sum[own]=__shfl_sync(0xffffffffu,denominator,0);
            if(lane==0)row_scales[row]=scale;
        }
        __syncthreads(); // Every score reader has finished; row scales are published.
        for(int i=tid;i<Rows*16;i+=Threads)scores[i]=row_scales[i/16];
        __syncthreads();
        Acc scales[Groups],product[Groups][4];
#pragma unroll
        for(int g=0;g<Groups;++g){wm::load_matrix_sync(scales[g],scores+(row_group+g)*16*16,16,wm::mem_row_major);
#pragma unroll
            for(int j=0;j<4;++j)wm::fill_fragment(product[g][j],0.0f);
        }
        for(int d=0;d<Keys;d+=16){
            wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major>a[Groups];
#pragma unroll
            for(int g=0;g<Groups;++g)wm::load_matrix_sync(a[g],probability+(row_group+g)*16*Keys+d,Keys);
#pragma unroll
            for(int j=0;j<4;++j){wm::fragment<wm::matrix_b,16,16,16,half,wm::row_major>b;
                wm::load_matrix_sync(b,v+size_t(key+d)*CacheStride+kv*Dim+column_warp*64+j*16,CacheStride);
#pragma unroll
                for(int g=0;g<Groups;++g)wm::mma_sync(product[g][j],a[g],b,product[g][j]);
            }
        }
#pragma unroll
        for(int g=0;g<Groups;++g){
#pragma unroll
            for(int j=0;j<4;++j){
#pragma unroll
                for(int i=0;i<running[g][j].num_elements;++i)running[g][j].x[i]=fmaf(running[g][j].x[i],scales[g].x[i],product[g][j].x[i]);
            }
        }
        __syncthreads();
    }
    if(lane==0){
#pragma unroll
        for(int own=0;own<Own;++own)row_scales[warp+own*Warps]=previous_sum[own];
    }
    __syncthreads();for(int i=tid;i<Rows*16;i+=Threads)scores[i]=row_scales[i/16];__syncthreads();
#pragma unroll
    for(int g=0;g<Groups;++g){Acc denominator;wm::load_matrix_sync(denominator,scores+(row_group+g)*16*16,16,wm::mem_row_major);
#pragma unroll
        for(int j=0;j<4;++j){
#pragma unroll
            for(int i=0;i<running[g][j].num_elements;++i)running[g][j].x[i]/=denominator.x[i];
        }
    }
    if(active==Rows){
#pragma unroll
        for(int g=0;g<Groups;++g){
#pragma unroll
            for(int j=0;j<4;++j)wm::store_matrix_sync(out+(size_t(qbase+(row_group+g)*16)*Heads+head)*Dim+column_warp*64+j*16,
                running[g][j],Heads*Dim,wm::mem_row_major);
        }
    }else{
        __syncthreads(); // Protect denominator loads before score-tail staging.
#pragma unroll
        for(int j=0;j<4;++j){
#pragma unroll
            for(int g=0;g<Groups;++g)wm::store_matrix_sync(scores+(row_group+g)*16*Keys+column_warp*16,running[g][j],Keys,wm::mem_row_major);
            __syncthreads();
            for(int i=tid;i<active*64;i+=Threads){const int row=i/64,col=i%64,d=(col/16)*64+j*16+col%16;
                out[(size_t(qbase+row)*Heads+head)*Dim+d]=scores[i];}
            __syncthreads();
        }
    }
#endif
}
AttentionRows32Resources resources(){
    const auto p=actual_sm75();
    AttentionRows32Resources r{32,128,0,0,0,0,0,shared_bytes<32>(),0,p.sharedMemPerBlockOptin};
    if(p.major!=7||p.minor!=5){r.reason_mask=64;return r;}
    cudaFuncAttributes a{};check(cudaFuncGetAttributes(&a,query_tile<32,128>));
    r.registers=a.numRegs;r.binary_version=a.binaryVersion;r.static_shared=a.sharedSizeBytes;r.local_bytes=a.localSizeBytes;
    if(r.static_shared+r.dynamic_shared>r.optin_limit)r.reason_mask|=1;
    if(r.local_bytes)r.reason_mask|=2;
    if(a.numRegs*128>p.regsPerBlock||a.maxThreadsPerBlock<128)r.reason_mask|=4;
    if(a.binaryVersion!=75)r.reason_mask|=16;
    if(!r.reason_mask){
        // 28,800B needs no opt-in on SM75. Respect actual default/opt-in limits.
        if(r.static_shared+r.dynamic_shared>p.sharedMemPerBlock){
            const auto error=cudaFuncSetAttribute(query_tile<32,128>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(r.dynamic_shared));
            if(error!=cudaSuccess){(void)cudaGetLastError();r.reason_mask|=32;}
        }
        if(!r.reason_mask){check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,query_tile<32,128>,128,r.dynamic_shared));if(!r.active_blocks)r.reason_mask|=8;}
    }
    return r;
}
}
AttentionRows32Resources attention_rows32_resources(){return resources();}
bool attention_rows32_prefill(const float* q,uint16_t* k,uint16_t* v,int first,int count,int capacity,float* out,void* stream){
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity))throw std::invalid_argument("unsupported Rows32 attention geometry");
    if(resources().reason_mask)return false; // no KV/output write before qualification
    const int width=first+count,padded=((width+63)/64)*64;const auto s=static_cast<cudaStream_t>(stream);
    if(padded>width)zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    query_tile<32,128><<<dim3(Heads,(count+31)/32),128,shared_bytes<32>(),s>>>(q,reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    check(cudaGetLastError());return true;
}
}
