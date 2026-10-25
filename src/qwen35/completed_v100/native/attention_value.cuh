#pragma once
#include <mma.h>
#include "attention_volta.cuh"

namespace strata::completed_v100::qwen35::native {
namespace {
// Uniform-CTA interface keeps the numerical partial boundaries independent
// of the hardware atom and of the reference kernel's thread/block layout.
struct PrefillValues {
#if __CUDA_ARCH__ == 700
    VoltaValueAccumulator value[2];
#else
    using HalfFragment=nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,__half>;
    HalfFragment value[2][4];
#endif
    __device__ void clear(){
        #pragma unroll
        for(int p=0;p<2;++p){
#if __CUDA_ARCH__ == 700
            value[p].clear();
#else
            #pragma unroll
            for(int f=0;f<4;++f)nvcuda::wmma::fill_fragment(value[p][f],__float2half(0));
#endif
        }
    }
    __device__ void store(__half *scratch,int warp,int lane){
        #pragma unroll
        for(int p=0;p<2;++p){
#if __CUDA_ARCH__ == 700
            value[p].store(scratch+p*4096,warp,lane);
#else
            #pragma unroll
            for(int f=0;f<4;++f)nvcuda::wmma::store_matrix_sync(scratch+p*4096+(warp*4+f)*16,
                value[p][f],256,nvcuda::wmma::mem_col_major);
#endif
        }
    }
    __device__ void rescale(const float *scales,__half *scratch,int warp,int tid){
#if __CUDA_ARCH__ == 700
        value[0].rescale(scales,warp);value[1].rescale(scales+16,warp);
#else
        store(scratch,warp,tid%32);__syncthreads();
        for(int i=tid;i<8192;i+=128)scratch[i]=__hmul(scratch[i],__float2half_rn(scales[i/4096*16+i%4096/256]));
        __syncthreads();
        #pragma unroll
        for(int p=0;p<2;++p){
            #pragma unroll
            for(int f=0;f<4;++f)nvcuda::wmma::load_matrix_sync(value[p][f],scratch+p*4096+(warp*4+f)*16,
                256,nvcuda::wmma::mem_col_major);
        }
        __syncthreads();
#endif
    }
    __device__ void accumulate(const __half *v,const __half *prob,int warp,int lane){
        #pragma unroll
        for(int p=0;p<2;++p){
#if __CUDA_ARCH__ == 700
            value[p].accumulate(v,prob+p*512,warp,lane,p);
#else
            #pragma unroll
            for(int key=0;key<32;key+=16){
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__half,nvcuda::wmma::col_major> b;
                nvcuda::wmma::load_matrix_sync(b,prob+p*512+key,32);
                #pragma unroll
                for(int f=0;f<4;++f){
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__half,nvcuda::wmma::col_major> a;
                    nvcuda::wmma::load_matrix_sync(a,v+key*256+(warp*4+f)*16,256);
                    nvcuda::wmma::mma_sync(value[p][f],a,b,value[p][f]);
                }
            }
#endif
        }
    }
};
}
}
