#include "gdn_delta_half.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <mma.h>
#include <cmath>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Pitch=136,Small=40;
struct Scan {
    // At the state-update phase both half matrices are dead; alias their combined storage as FP32.
    __align__(32) __half matrix[2][32*Pitch],k[32*Pitch],b[32*Small],delta[32*Small];
    __align__(32) float temporary[32*Small];
    float2 factor[32];
};
using Accumulator=wmma::fragment<wmma::accumulator,16,16,16,float>;
__device__ void state_product(Scan &p,int tile,Accumulator &c){
    const int row=tile/2*16,value=tile%2*16;wmma::fill_fragment(c,0.f);
    #pragma unroll
    for(int d=0;d<128;d+=16){
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
        wmma::load_matrix_sync(a,p.matrix[1]+row*Pitch+d,Pitch);wmma::load_matrix_sync(b,p.matrix[0]+value*Pitch+d,Pitch);
        wmma::mma_sync(c,a,b,c);
    }
}
__global__ void scan(GdnInputs x,GdnShape s,GdnFloatValueWork work,float *out,float *final_state,float scale){
    extern __shared__ __align__(32) unsigned char arena[];auto &p=*reinterpret_cast<Scan *>(arena);
    const int tid=threadIdx.x,warp=tid/32,head=blockIdx.y,value_first=int(blockIdx.x)*32;
    const int query=head%s.query_heads,chunks=(s.tokens+31)/32;
    float state[16];
    #pragma unroll
    for(int r=0;r<16;++r){const int i=tid+r*256;state[r]=x.state[(size_t(head)*128+value_first+i/128)*128+i%128];}
    for(int chunk=0;chunk<chunks;++chunk){
        const size_t block=size_t(head)*chunks+chunk;const int first=chunk*32;
        if(tid<32){const float g=work.prefix[block*32+tid],end=work.prefix[block*32+31];p.factor[tid]=make_float2(expf(g),expf(end-g));}
        __syncthreads();const float decay=p.factor[31].x;
        #pragma unroll
        for(int r=0;r<16;++r){const int i=tid+r*256;p.matrix[0][(i/128)*Pitch+i%128]=__float2half_rn(state[r]);}
        for(int i=tid;i<32*128;i+=256){const int t=i/128,d=i%128,actual=first+t;float k=0;
            if(actual<s.tokens)k=x.k[(size_t(actual)*s.query_heads+query)*128+d]*p.factor[t].y;
            p.k[t*Pitch+d]=__float2half_rn(k);p.matrix[1][t*Pitch+d]=work.w[block*32*128+i];}
        for(int i=tid;i<32*32;i+=256)p.b[(i/32)*Small+i%32]=work.b[block*32*32+i];
        __syncthreads();
        if(warp<4){Accumulator c;state_product(p,warp,c);wmma::store_matrix_sync(p.temporary+(warp/2)*16*Small+(warp%2)*16,c,Small,wmma::mem_row_major);}
        __syncthreads();
        for(int i=tid;i<32*32;i+=256){const int t=i/32,v=i%32;
            const size_t at=block*32*128+t*128+value_first+v;
            const float full=__fsub_rn(work.u[at],p.temporary[t*Small+v]);
            const __half high=__float2half_rn(full);p.delta[t*Small+v]=high;
            // Keep work.u in FP32 through cancellation; round the resulting delta once.
        }
        for(int i=tid;i<32*128;i+=256){const int t=i/128,d=i%128,actual=first+t;float q=0;
            if(actual<s.tokens)q=x.q[(size_t(actual)*s.query_heads+query)*128+d]*p.factor[t].x;
            p.matrix[1][t*Pitch+d]=__float2half_rn(q);}
        __syncthreads();
        if(warp<4){
            const int row=warp/2*16,value=warp%2*16;Accumulator c;state_product(p,warp,c);
            #pragma unroll
            for(int d=0;d<32;d+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
                wmma::load_matrix_sync(a,p.b+row*Small+d,Small);wmma::load_matrix_sync(b,p.delta+d*Small+value,Small);wmma::mma_sync(c,a,b,c);
            }
            wmma::store_matrix_sync(p.temporary+row*Small+value,c,Small,wmma::mem_row_major);
        }
        __syncthreads();
        for(int i=tid;i<32*32;i+=256){const int t=first+i/32;if(t<s.tokens)out[(size_t(t)*s.value_heads+head)*128+value_first+i%32]=p.temporary[(i/32)*Small+i%32]*scale;}
        __syncthreads();
        auto *update=reinterpret_cast<float *>(p.matrix);
        for(int tile=warp;tile<16;tile+=8){
            const int value=tile/8*16,key=tile%8*16;Accumulator c;wmma::fill_fragment(c,0.f);
            #pragma unroll
            for(int d=0;d<32;d+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
                wmma::load_matrix_sync(a,p.delta+d*Small+value,Small);wmma::load_matrix_sync(b,p.k+d*Pitch+key,Pitch);wmma::mma_sync(c,a,b,c);
            }
            wmma::store_matrix_sync(update+value*Pitch+key,c,Pitch,wmma::mem_row_major);
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<16;++r){const int i=tid+r*256;state[r]=fmaf(decay,state[r],update[(i/128)*Pitch+i%128]);}
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<16;++r){const int i=tid+r*256;final_state[(size_t(head)*128+value_first+i/128)*128+i%128]=state[r];}
}
}
GdnDeltaResources gdn_delta_half_resources(){
    cudaFuncAttributes attr{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&attr,scan),"delta half resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,scan,256,sizeof(Scan)),"delta half occupancy");
    return {attr.numRegs,blocks,sizeof(Scan),attr.localSizeBytes};
}

void gdn_delta_half_configure(){check_cuda(cudaFuncSetAttribute(scan,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Scan)),"tileGDN scan shared");}
void gdn_delta_half(GdnInputs x,GdnShape s,GdnFloatValueWork work,float *out,float *state,cudaStream_t stream){
    if(!x.q||!x.k||!x.v||!x.log_decay||!x.beta||!x.state||!out||!state||!work.w||!work.u||!work.b||!work.prefix||
        s.key_dim!=128||s.value_dim!=128||s.query_heads<1||s.value_heads<1||s.value_heads%s.query_heads||s.tokens<1||s.tokens>8192)
        throw std::invalid_argument("tileGDN supported geometry/storage");
    scan<<<dim3(4,s.value_heads),256,sizeof(Scan),stream>>>(x,s,work,out,state,1.f/std::sqrt(128.f));
    check_cuda(cudaGetLastError(),"tileGDN register state scan");
}
}
