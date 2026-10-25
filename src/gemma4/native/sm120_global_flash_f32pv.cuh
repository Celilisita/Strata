#pragma once
// The caller already includes the SHA-pinned existing mma.cuh. Only its
// F16-input/F32-accumulator ldmatrix/MMA primitives are reused, not a runtime.
#include "sm120_global_flash_f32pv.hpp"
#include <cfloat>
namespace strata::gemma4::native::f32pv_detail {
namespace p=strata::qwen35::native;
using namespace ggml_cuda_mma;
using A=tile<16,8,half2>;
using B=tile<8,8,half2>;
using C=tile<16,8,float>;
static_assert(C::ne==4&&A::ne==4&&B::ne==2,"Pinned SM120 fragment ABI");
__device__ inline float maximum_rows(float a){
    a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,4));
    a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,8));
    return fmaxf(a,__shfl_xor_sync(0xffffffff,a,16));
}
__device__ inline float sum_rows(float a){
    a+=__shfl_xor_sync(0xffffffff,a,4);a+=__shfl_xor_sync(0xffffffff,a,8);
    return a+__shfl_xor_sync(0xffffffff,a,16);
}
__global__ __launch_bounds__(Threads,1) void kernel(const float* q,const half* key,const half* value,
        const half* mask,float* output,int tokens,int keys){
#if defined(TURING_MMA_AVAILABLE)
    const int lane=threadIdx.x,warp=threadIdx.y,thread=warp*32+lane;
    const int kv=blockIdx.x%4,first_query=int(blockIdx.x/4)*QueryTile;
    // Warp pairs own one query token, eight GQA heads and disjoint16-key
    // halves. Fragment j is the GQA head; fragment i is key (QK) or D (PV).
    const int query=first_query+warp/2,part=warp%2;
    extern __shared__ __align__(16) half2 arena[];
    half2* const Q=arena;half2* const KV=Q+32*Stride;
    half* const P=reinterpret_cast<half*>(KV+32*Stride);
    for(int i=thread;i<32*(Dim/2);i+=Threads){const int column=i/(Dim/2),d2=i%(Dim/2);
        const int t=first_query+column/Gqa,h=kv*Gqa+column%Gqa;half2 v=__float2half2_rn(0.f);
        if(t<tokens){const auto at=(std::size_t(t)*32+h)*Dim+2*d2;
            v=__floats2half2_rn(q[at],q[at+1]);}
        Q[column*Stride+d2]=v;
    }
    __syncthreads();
    //32 fragments*4F32/thread=128 accumulator registers. QueryTile8 would
    //double this base before QK, address and softmax registers are counted.
    C accumulator[Dim/16];float top[2]={-FLT_MAX/2,-FLT_MAX/2},denominator[2]={0,0};
    for(int first=0;first<keys;first+=KeyTile){
        for(int i=thread;i<KeyTile*(Dim/2);i+=Threads){const int row=i/(Dim/2),d2=i%(Dim/2);
            KV[row*Stride+d2]=reinterpret_cast<const half2*>(key)[(std::size_t(kv)*keys+first+row)*(Dim/2)+d2];}
        __syncthreads();C score;
        #pragma unroll
        for(int d=0;d<Dim;d+=16){A a;B b;
            load_ldmatrix(a,KV+part*16*Stride+d/2,Stride);
            load_ldmatrix(b,Q+(warp/2)*Gqa*Stride+d/2,Stride);mma(score,a,b);}
        float newest[2]={top[0],top[1]};
        for(int l=0;l<C::ne;++l){const int k=first+part*16+C::get_i(l);
            score.x[l]+=query<tokens?__half2float(mask[std::size_t(query)*keys+k]):-INFINITY;
            newest[l%2]=fmaxf(newest[l%2],score.x[l]+FATTN_KQ_MAX_OFFSET);}
        newest[0]=maximum_rows(newest[0]);newest[1]=maximum_rows(newest[1]);
        float scale[2],row_sum[2]={0,0};
        for(int s=0;s<2;++s){const float change=top[s]-newest[s];
            scale[s]=change>=SOFTMAX_FTZ_THRESHOLD?expf(change):0.f;top[s]=newest[s];}
        for(int l=0;l<C::ne;++l){const float probability=expf(score.x[l]-top[l%2]);row_sum[l%2]+=probability;
            P[(warp*Gqa+C::get_j(l))*16+C::get_i(l)]=__float2half_rn(probability);}
        for(int s=0;s<2;++s)denominator[s]=fmaf(denominator[s],scale[s],sum_rows(row_sum[s]));
        // K is dead. All warps finish QK before the one shared tile holds V.
        __syncthreads();
        for(int i=thread;i<KeyTile*(Dim/2);i+=Threads){const int row=i/(Dim/2),d2=i%(Dim/2);
            KV[row*Stride+d2]=reinterpret_cast<const half2*>(value)[(std::size_t(kv)*keys+first+row)*(Dim/2)+d2];}
        __syncthreads();B probability;
        load_ldmatrix(probability,reinterpret_cast<const half2*>(P)+warp*Gqa*8,8);
        #pragma unroll
        for(int block=0;block<Dim/16;++block){
            for(int l=0;l<C::ne;++l)accumulator[block].x[l]*=scale[l%2];
            A v;load_ldmatrix_trans(v,KV+part*16*Stride+block*8,Stride);
            mma(accumulator[block],v,probability);
        }
        __syncthreads();
    }
    // Q and KV are dead. Reuse their shared storage for one16-D slab and
    // metadata, retaining F32 through the warp-pair merge and final store.
    float* const partial=reinterpret_cast<float*>(Q);
    float* const meta=reinterpret_cast<float*>(KV);
    if(lane<4)for(int s=0;s<2;++s){const int head=lane*2+s;
        meta[(warp*Gqa+head)*2]=top[s];meta[(warp*Gqa+head)*2+1]=denominator[s];}
    __syncthreads();
    #pragma unroll
    for(int block=0;block<Dim/16;++block){
        for(int l=0;l<C::ne;++l)partial[(warp*Gqa+C::get_j(l))*16+C::get_i(l)]=accumulator[block].x[l];
        __syncthreads();
        if(part==0&&query<tokens)for(int l=0;l<C::ne;++l){const int h=C::get_j(l),d=block*16+C::get_i(l);
            const int a=(warp*Gqa+h)*2,b=((warp+1)*Gqa+h)*2;const float maximum=fmaxf(meta[a],meta[b]);
            const float sa=expf(meta[a]-maximum),sb=expf(meta[b]-maximum);
            const float den=fmaf(meta[a+1],sa,meta[b+1]*sb);
            const float num=fmaf(partial[(warp*Gqa+h)*16+C::get_i(l)],sa,partial[((warp+1)*Gqa+h)*16+C::get_i(l)]*sb);
            output[(std::size_t(query)*32+kv*Gqa+h)*Dim+d]=num/den;
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
#endif
}
inline void prepare(){
    p::check_cuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),cudaFuncAttributeMaxDynamicSharedMemorySize,int(Shared)),"Gemma F32PV shared preparation");
    cudaFuncAttributes attributes{};p::check_cuda(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(kernel)),"Gemma F32PV actual resources");
    std::fprintf(stderr,"GEMMA_GLOBAL_FLASH_PREP dtype=F32_PV_accum query_tile=4 threads=%d dynamic_shared=%zu registers=%d local_bytes=%zu no_forward=1\n",Threads,Shared,attributes.numRegs,attributes.localSizeBytes);
}
inline void launch(const float* q,const void* k,const void* v,const void* mask,float* out,int tokens,int keys,cudaStream_t stream){
    kernel<<<dim3(((tokens+QueryTile-1)/QueryTile)*4),dim3(32,Warps),Shared,stream>>>(q,static_cast<const half*>(k),static_cast<const half*>(v),static_cast<const half*>(mask),out,tokens,keys);
    p::check_cuda(cudaGetLastError(),"Gemma F32PV fixed4x8 complete leaf");
}
}
