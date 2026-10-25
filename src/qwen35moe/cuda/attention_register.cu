#include "attention_register.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
namespace wm=nvcuda::wmma;
constexpr int Rows=16,Keys=64,Dim=256,Heads=16,CacheStride=512;
__device__ float maximum(float x) {
    for(int d=16;d;d>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,d));return x;
}
__device__ float total(float x) {
    for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffffu,x,d);return x;
}
__global__ void zero_tail(half* k,half* v,int width,int cells) {
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<cells){k[size_t(width)*CacheStride+i]=__float2half_rn(0);v[size_t(width)*CacheStride+i]=__float2half_rn(0);}
}
struct Shared {
    half query[Rows*Dim];float scores[Rows*Keys];half probability[Rows*Keys];float scales[Rows];
};
static_assert(sizeof(Shared)==14400);
__global__ __launch_bounds__(128,3) void register_output(const float* q,const half* k,const half* v,
    int first,int count,float* out) {
#if __CUDA_ARCH__ == 750
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,head=blockIdx.x,kv=head/8,qbase=blockIdx.y*Rows;
    const int active=min(Rows,count-qbase),width=first+qbase+active;
    __shared__ __align__(32) Shared s;
    using Acc=wm::fragment<wm::accumulator,16,16,16,float>;
    Acc running[4];
#pragma unroll
    for(int j=0;j<4;++j)wm::fill_fragment(running[j],0.0f);
    float previous_max[4],previous_sum[4];
#pragma unroll
    for(int own=0;own<4;++own){previous_max[own]=-CUDART_INF_F;previous_sum[own]=0;}
    for(int i=tid;i<Rows*Dim;i+=128){
        const int row=i/Dim,d=i%Dim;const float value=row<active?q[(size_t(qbase+row)*Heads+head)*Dim+d]:0;
        s.query[i]=__hmul(__float2half_rn(value*0.0625f),__float2half_rn(0.25f));
    }
    __syncthreads();
    for(int key=0;key<width;key+=Keys){
        {
            wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major>a;
            wm::fragment<wm::matrix_b,16,16,16,half,wm::col_major>b;
            Acc c;wm::fill_fragment(c,0.0f);
            for(int d=0;d<Dim;d+=16){
                wm::load_matrix_sync(a,s.query+d,Dim);
                wm::load_matrix_sync(b,k+size_t(key+warp*16)*CacheStride+kv*Dim+d,CacheStride);
                wm::mma_sync(c,a,b,c);
            }
            wm::store_matrix_sync(s.scores+warp*16,c,Keys,wm::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for(int own=0;own<4;++own){
            const int row=warp+own*4,end=first+qbase+row+1;const bool valid=row<active;
            const float s0=valid&&key+lane<end?s.scores[row*Keys+lane]*4.0f:-CUDART_INF_F;
            const float s1=valid&&key+lane+32<end?s.scores[row*Keys+lane+32]*4.0f:-CUDART_INF_F;
            const float m=valid?fmaxf(previous_max[own],maximum(fmaxf(s0,s1))):0;
            const float p0=valid?expf(s0-m):0,p1=valid?expf(s1-m):0;
            const float scale=valid?expf(previous_max[own]-m):0;
            const float sum=total(p0+p1);
            const float denominator=valid?previous_sum[own]*scale+sum:1;
            s.probability[row*Keys+lane]=__float2half_rn(p0);
            s.probability[row*Keys+lane+32]=__float2half_rn(p1);
            previous_max[own]=__shfl_sync(0xffffffffu,m,0);
            previous_sum[own]=__shfl_sync(0xffffffffu,denominator,0);
            if(lane==0)s.scales[row]=scale;
        }
        __syncthreads(); // Every softmax score reader has finished; scores are now dead.
        for(int i=tid;i<Rows*16;i+=128)s.scores[i]=s.scales[i/16];
        __syncthreads();
        Acc scale;wm::load_matrix_sync(scale,s.scores,16,wm::mem_row_major);
        wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major>a;
        wm::fragment<wm::matrix_b,16,16,16,half,wm::row_major>b;
        Acc product[4];
#pragma unroll
        for(int j=0;j<4;++j)wm::fill_fragment(product[j],0.0f);
        for(int d=0;d<Keys;d+=16){
            wm::load_matrix_sync(a,s.probability+d,Keys);
#pragma unroll
            for(int j=0;j<4;++j){
                wm::load_matrix_sync(b,v+size_t(key+d)*CacheStride+kv*Dim+warp*64+j*16,CacheStride);
                wm::mma_sync(product[j],a,b,product[j]);
            }
        }
#pragma unroll
        for(int j=0;j<4;++j){
#pragma unroll
            for(int i=0;i<running[j].num_elements;++i)
                running[j].x[i]=fmaf(running[j].x[i],scale.x[i],product[j].x[i]);
        }
        __syncthreads(); // All warps have loaded scale before the next QK overwrites scores.
    }
    if(lane==0){
#pragma unroll
        for(int own=0;own<4;++own)s.scales[warp+own*4]=previous_sum[own];
    }
    __syncthreads();
    for(int i=tid;i<Rows*16;i+=128)s.scores[i]=s.scales[i/16];
    __syncthreads();
    Acc denominator;wm::load_matrix_sync(denominator,s.scores,16,wm::mem_row_major);
#pragma unroll
    for(int j=0;j<4;++j){
#pragma unroll
        for(int i=0;i<running[j].num_elements;++i)running[j].x[i]/=denominator.x[i];
    }
    if(active==Rows){
#pragma unroll
        for(int j=0;j<4;++j)wm::store_matrix_sync(out+(size_t(qbase)*Heads+head)*Dim+warp*64+j*16,
            running[j],Heads*Dim,wm::mem_row_major);
    }else{
        __syncthreads(); // All denominator fragments are in registers before score staging.
#pragma unroll
        for(int j=0;j<4;++j){
            wm::store_matrix_sync(s.scores+warp*16,running[j],64,wm::mem_row_major);
            __syncthreads();
            for(int i=tid;i<active*64;i+=128){
                const int row=i/64,col=i%64,d=(col/16)*64+j*16+col%16;
                out[(size_t(qbase+row)*Heads+head)*Dim+d]=s.scores[i];
            }
            __syncthreads();
        }
    }
#endif
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void actual_sm75(){int ordinal=0;cudaDeviceProp p{};check(cudaGetDevice(&ordinal));check(cudaGetDeviceProperties(&p,ordinal));
    if(p.major!=7||p.minor!=5)throw std::invalid_argument("register attention qualification requires actual SM75");}
}
void attention_register_prefill(const float* q,uint16_t* k,uint16_t* v,int first,int count,int capacity,float* out,void* stream){
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity))throw std::invalid_argument("unsupported register attention geometry");
    actual_sm75();const int width=first+count,padded=((width+63)/64)*64;auto s=static_cast<cudaStream_t>(stream);
    if(padded>width)zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    register_output<<<dim3(Heads,(count+Rows-1)/Rows),128,0,s>>>(q,reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    check(cudaGetLastError());
}
AttentionTcResources attention_register_resources(){actual_sm75();cudaFuncAttributes a{};check(cudaFuncGetAttributes(&a,register_output));
    if(a.binaryVersion!=75)throw std::runtime_error("register attention requires native75 cubin");return{a.numRegs,a.sharedSizeBytes,a.localSizeBytes};}
int attention_register_active_blocks(){actual_sm75();int blocks=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,register_output,128,0));return blocks;}
}
