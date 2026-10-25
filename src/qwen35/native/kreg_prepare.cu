#include "kreg.hpp"
#include "kreg_cache.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <mma.h>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Pitch=136,Small=40,Features=264;
template<bool Grouped> struct Prep {
    __align__(32) __half qk[2*32*Pitch],input[32*Features];
    __align__(32) float score[32*Small],inverse[32*Small];
    float prefix[Grouped?96:32],beta[32],beta_exp[32];
};
using Accumulator=wmma::fragment<wmma::accumulator,16,16,16,float>;
template<class P> __device__ void dots(P &p,const __half *left,int warp){
    const int row=warp/2*16,col=warp%2*16;
    Accumulator c;wmma::fill_fragment(c,0.f);
    #pragma unroll
    for(int d=0;d<128;d+=16){
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
        wmma::load_matrix_sync(a,left+row*Pitch+d,Pitch);wmma::load_matrix_sync(b,p.qk+32*Pitch+col*Pitch+d,Pitch);
        wmma::mma_sync(c,a,b,c);
    }
    wmma::store_matrix_sync(p.score+row*Small+col,c,Small,wmma::mem_row_major);
}
template<class P> __device__ __forceinline__ void parameters(P &p,GdnInputs x,GdnShape s,
        int head,int first,int warp,int lane,float *prefix,size_t block,GdnFloatValueWork out){
    if(!warp){
        const int t=first+lane;float g=t<s.tokens?x.log_decay[size_t(t)*s.value_heads+head]:0;
        for(int offset=1;offset<32;offset*=2){const float before=__shfl_up_sync(0xffffffff,g,offset);if(lane>=offset)g+=before;}
        prefix[lane]=g;p.beta[lane]=t<s.tokens?x.beta[size_t(t)*s.value_heads+head]:0;
        p.beta_exp[lane]=p.beta[lane]*expf(g);out.prefix[block*32+lane]=g;
    }
}
template<class P> __device__ __forceinline__ float weighted_score(P &p,const float *prefix,int t,int j,float score){
    return j<t?p.beta[t]*expf(prefix[t]-prefix[j])*score:0;
}
template<bool Cached,class P> __device__ __forceinline__ void solve_rhs(P &p,GdnInputs x,GdnShape s,
        GdnFloatValueWork out,int head,int query,int first,size_t block,int tid,int warp,int lane,const kreg_detail::PackedK &cache){
    // Invert unit lower triangular I+A once, rather than solving 256 columns separately.
    if(tid<32){
        float column[32];
        #pragma unroll
        for(int t=0;t<32;++t){
            float value=tid==t?1.f:0.f;
            #pragma unroll
            for(int j=0;j<t;++j)value=fmaf(-p.score[t*Small+j],column[j],value);
            column[t]=value;p.inverse[t*Small+tid]=value;
        }
    }
    // Cross-lane consumers start only after every private column is published.
    __syncthreads();
    auto *lower=reinterpret_cast<__half *>(p.score),*lower_low=lower+32*Small;
    for(int i=tid;i<32*32;i+=128){const int at=(i/32)*Small+i%32;const float value=p.inverse[at];
        lower[at]=__float2half_rn(value);lower_low[at]=__float2half_rn(value-__half2float(lower[at]));}
    // Q/K are dead until B. Their combined storage holds the RHS low component.
    auto *input_low=p.qk;
    if constexpr(Cached)kreg_detail::rhs<0>(p,cache,x,s,head,first,tid,input_low);
    else {
    for(int i=tid;i<32*256;i+=128){const int t=i/256,col=i%256,actual=first+t;float value=0;
        if(col<128&&actual<s.tokens)value=p.beta_exp[t]*__half2float(__float2half_rn(x.k[(size_t(actual)*s.query_heads+query)*128+col]));
        else if(actual<s.tokens)value=p.beta[t]*x.v[(size_t(actual)*s.value_heads+head)*128+col-128];
        p.input[t*Features+col]=__float2half_rn(value);
        input_low[t*Features+col]=__float2half_rn(value-__half2float(p.input[t*Features+col]));
    }
    }
    __syncthreads();
    // The FP32 inverse is now dead and supplies independent per-warp accumulator stores.
    for(int tile=warp;tile<32;tile+=4){
        const int row=tile/16*16,col=tile%16*16;Accumulator c;wmma::fill_fragment(c,0.f);
        #pragma unroll
        for(int d=0;d<32;d+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> al;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bl;
            wmma::load_matrix_sync(a,lower+row*Small+d,Small);wmma::load_matrix_sync(b,p.input+d*Features+col,Features);
            wmma::mma_sync(c,a,b,c);
            wmma::load_matrix_sync(al,lower_low+row*Small+d,Small);wmma::load_matrix_sync(bl,input_low+d*Features+col,Features);
            wmma::mma_sync(c,a,bl,c);wmma::mma_sync(c,al,b,c);
        }
        float *temporary=p.inverse+warp*256;wmma::store_matrix_sync(temporary,c,16,wmma::mem_row_major);__syncwarp();
        for(int i=lane;i<256;i+=32){const int t=row+i/16,feature=col+i%16;
            const int d=feature%128;const size_t at=block*32*128+t*128+d;
            if(feature<128)out.w[at]=__float2half_rn(temporary[i]);
            else out.u[at]=temporary[i];}

        __syncwarp();
    }
}
template<bool Grouped> __global__ void prepare(GdnInputs x,GdnShape s,GdnFloatValueWork out){
    extern __shared__ __align__(32) unsigned char arena[];auto &p=*reinterpret_cast<Prep<Grouped> *>(arena);
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,first=int(blockIdx.x)*32;
    const int chunks=(s.tokens+31)/32;
    const int query=Grouped?int(blockIdx.y):int(blockIdx.y)%s.query_heads;
    kreg_detail::PackedK cache;
    if constexpr(Grouped)kreg_detail::load<0>(cache,p,x,s,query,first,tid);
    else {
    for(int i=tid;i<32*128;i+=128){
        const int t=first+i/128,d=i%128;float k=0;
        if(t<s.tokens)k=x.k[(size_t(t)*s.query_heads+query)*128+d];
        p.qk[32*Pitch+(i/128)*Pitch+d]=__float2half_rn(k);
    }
    }
    if constexpr(Grouped){
        __syncthreads();dots(p,p.qk+32*Pitch,warp);__syncthreads();
        float raw[8];
        #pragma unroll
        for(int r=0;r<8;++r){const int i=tid+r*128;raw[r]=p.score[(i/32)*Small+i%32];}
        #pragma unroll 1
        for(int part=0;part<3;++part){
            const int head=query+part*s.query_heads;const size_t block=size_t(head)*chunks+blockIdx.x;
            float *prefix=p.prefix+part*32;
            parameters(p,x,s,head,first,warp,lane,prefix,block,out);__syncthreads();
            #pragma unroll
            for(int r=0;r<8;++r){const int i=tid+r*128,t=i/32,j=i%32;
                p.score[t*Small+j]=weighted_score(p,prefix,t,j,raw[r]);}
            __syncthreads();solve_rhs<Grouped>(p,x,s,out,head,query,first,block,tid,warp,lane,cache);__syncthreads();
        }
    }else{
        // Preserve the current control's parameter-before-KK order and resources.
        const int head=blockIdx.y;const size_t block=size_t(head)*chunks+blockIdx.x;
        parameters(p,x,s,head,first,warp,lane,p.prefix,block,out);
        __syncthreads();dots(p,p.qk+32*Pitch,warp);__syncthreads();
        for(int i=tid;i<32*32;i+=128){const int t=i/32,j=i%32;
            p.score[t*Small+j]=weighted_score(p,p.prefix,t,j,p.score[t*Small+j]);}
        __syncthreads();solve_rhs<Grouped>(p,x,s,out,head,query,first,block,tid,warp,lane,cache);__syncthreads();
    }
    if constexpr(Grouped)kreg_detail::qk<0>(p,cache,x,s,query,first,tid);
    else {
    for(int i=tid;i<32*128;i+=128){const int t=first+i/128,d=i%128;float q=0,k=0;
        if(t<s.tokens){const size_t at=(size_t(t)*s.query_heads+query)*128+d;q=x.q[at];k=x.k[at];}
        p.qk[(i/128)*Pitch+d]=__float2half_rn(q);p.qk[32*Pitch+(i/128)*Pitch+d]=__float2half_rn(k);}
    }
    __syncthreads();dots(p,p.qk,warp);__syncthreads();
    #pragma unroll
    for(int part=0;part<(Grouped?3:1);++part){
        const int head=Grouped?query+part*s.query_heads:int(blockIdx.y);
        const size_t block=size_t(head)*chunks+blockIdx.x;const float *prefix=p.prefix+part*32;
        for(int i=tid;i<32*32;i+=128){const int t=i/32,j=i%32;
            out.b[block*32*32+i]=__float2half_rn(j<=t?expf(prefix[t]-prefix[j])*p.score[t*Small+j]:0);}
    }
}
template<bool Grouped> GdnPrepareResources resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,prepare<Grouped>),"GDN prepare attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,prepare<Grouped>,128,sizeof(Prep<Grouped>)),"GDN prepare occupancy");
    return {a.numRegs,blocks,sizeof(Prep<Grouped>),a.localSizeBytes};
}
}
void kreg_prepare_configure(){
    check_cuda(cudaFuncSetAttribute(prepare<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Prep<true>)),"K-register grouped prepare shared");
}
void kreg_prepare(GdnInputs x,GdnShape s,GdnFloatValueWork work,cudaStream_t stream){
    if(!use_gdn_qk_group(s)){gdn_register_prepare(x,s,work,stream);return;}
    prepare<true><<<dim3((s.tokens+31)/32,s.query_heads),128,sizeof(Prep<true>),stream>>>(x,s,work);
    check_cuda(cudaGetLastError(),"K-register grouped prepare");
}
GdnRegisterResources kreg_prepare_resources(){return {gdn_register_prepare_resources().legacy,resources<true>()};}
}
