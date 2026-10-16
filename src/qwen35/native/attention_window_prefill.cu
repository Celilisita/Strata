#include "strata/qwen35/native/attention_window.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <mma.h>
#include <cmath>
#include <algorithm>
#include <math_constants.h>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Dim=256,Keys=64,Pitch=Dim+16,ProbPitch=Keys+16,ScorePitch=Keys+8;
template<int Groups>struct Keys64QueryTile {
    static constexpr int Rows=16*Groups;
    __align__(32) __half q[Rows*Pitch],kv[Keys*Pitch],prob[2*Rows*ProbPitch];
    __align__(32) float score[Rows*ScorePitch];
    float maximum[Rows],denominator[Rows],adjust[Rows],seam_maximum[Rows],seam_scale[Rows];
};
__device__ float sum_warp(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float max_warp(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
// Issue independent global vector loads before their dependent shared stores.
// The fence only constrains compiler scheduling; it emits no GPU barrier.
template<int Threads>__device__ __forceinline__ void load_half(const __half *x,__half *tile,int first,int head,int heads,int valid){
    constexpr int Vectors=Keys*Dim/8,Chunks=(Vectors+Threads-1)/Threads;
    #pragma unroll
    for(int batch=0;batch<(Chunks+2)/3;++batch){
        uint4 pending[3];
        #pragma unroll
        for(int part=0;part<3;++part){
            const int chunk=batch*3+part,i=threadIdx.x+chunk*Threads,token=first+i/32,d=(i%32)*8;
            pending[part]=uint4{};
            if(chunk<Chunks&&i<Vectors&&token<valid)
                pending[part]=*reinterpret_cast<const uint4 *>(x+(size_t(token)*heads+head)*Dim+d);
        }
        asm volatile("" ::: "memory");
        #pragma unroll
        for(int part=0;part<3;++part){
            const int chunk=batch*3+part,i=threadIdx.x+chunk*Threads;
            if(chunk<Chunks&&i<Vectors)
                *reinterpret_cast<uint4 *>(tile+(i/32)*Pitch+(i%32)*8)=pending[part];
        }
    }
}
template<int Groups>__device__ void score_tile(Keys64QueryTile<Groups> &t,int warp){
    if(warp>=(Keys/16)*Groups)return;
    const int query=warp/(Keys/16),key=warp%(Keys/16);
    wmma::fragment<wmma::accumulator,16,16,16,float> dots;wmma::fill_fragment(dots,0.f);
    #pragma unroll
    for(int d=0;d<Dim;d+=16){
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
        wmma::load_matrix_sync(a,t.q+query*16*Pitch+d,Pitch);
        wmma::load_matrix_sync(b,t.kv+key*16*Pitch+d,Pitch);
        wmma::mma_sync(dots,a,b,dots);
    }
    wmma::store_matrix_sync(t.score+query*16*ScorePitch+key*16,dots,ScorePitch,wmma::mem_row_major);
}
template<int Groups>__global__ void keys64_kernel(const float *query,const __half *key,const __half *value,
                                         float *out,PrefillAttentionShape s,int window){
    constexpr int Rows=16*Groups,Threads=128*Groups,Warps=4*Groups;
    extern __shared__ __align__(32) unsigned char arena[];
    auto &t=*reinterpret_cast<Keys64QueryTile<Groups> *>(arena);
    if(s.graph_start){s.query_start=*s.graph_start;s.valid=s.query_start+s.tokens;}
    const int head=blockIdx.x,first_query=int(blockIdx.y)*Rows,kh=head/(s.query_heads/s.kv_heads);
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=threadIdx.x/128,local=threadIdx.x%128;
    const int end=min(s.valid,s.query_start+min(s.tokens,first_query+Rows));
    for(int i=threadIdx.x;i<Rows*Dim;i+=Threads){
        const int row=first_query+i/Dim;
        t.q[(i/Dim)*Pitch+i%Dim]=row<s.tokens?__hmul(__float2half_rn(query[(size_t(row)*s.query_heads+head)*Dim+i%Dim]),
                                  __float2half_rn(s.scale)):__float2half(0);
    }
    if(threadIdx.x<Rows){t.maximum[threadIdx.x]=-CUDART_INF_F;t.denominator[threadIdx.x]=0;
        t.seam_maximum[threadIdx.x]=-CUDART_INF_F;}
    // Load query-column tags through the same accumulator type. This derives
    // its architecture-specific element mapping without a lane/layout table.
    using Fragment=wmma::fragment<wmma::accumulator,16,16,16,float>;
    static_assert(Fragment::num_elements<=8,"packed accumulator tag capacity");
    auto *tags=reinterpret_cast<float *>(t.kv);
    for(int i=threadIdx.x;i<256;i+=Threads)tags[i]=float(i/16);
    __syncthreads();Fragment tag;wmma::load_matrix_sync(tag,tags,16,wmma::mem_col_major);
    unsigned columns=0;
    #pragma unroll
    for(int i=0;i<Fragment::num_elements;++i)columns|=unsigned(tag.x[i])<<(4*i);
    Fragment result[4],history[4];
    #pragma unroll
    for(int f=0;f<4;++f){wmma::fill_fragment(result[f],0.f);wmma::fill_fragment(history[f],0.f);}
    __syncthreads();
    for(int first=max(0,s.query_start+first_query-window+1)/Keys*Keys;first<end;first+=Keys){
        load_half<Threads>(key,t.kv,first,kh,s.kv_heads,s.valid);__syncthreads();
        score_tile(t,warp);__syncthreads();
        for(int row=warp;row<Rows;row+=Warps){
            const int token0=first+lane,token1=token0+32,boundary=s.query_start+first_query+row;
            const bool query_active=first_query+row<s.tokens;
            const bool active0=query_active&&token0<end&&token0<=boundary&&token0>=boundary-window+1;
            const bool active1=query_active&&token1<end&&token1<=boundary&&token1>=boundary-window+1;
            const float score0=active0?t.score[row*ScorePitch+lane]:-CUDART_INF_F;
            const float score1=active1?t.score[row*ScorePitch+lane+32]:-CUDART_INF_F;
            const float next=fmaxf(t.maximum[row],max_warp(fmaxf(score0,score1)));
            const float adjust=isfinite(t.maximum[row])?__expf(t.maximum[row]-next):0.f;
            const float p0=active0?__expf(score0-next):0.f,p1=active1?__expf(score1-next):0.f;
            const __half high0=__float2half_rn(p0),high1=__float2half_rn(p1);
            t.prob[row*ProbPitch+lane]=high0;t.prob[row*ProbPitch+lane+32]=high1;
            t.prob[Rows*ProbPitch+row*ProbPitch+lane]=__float2half_rn(p0-__half2float(high0));
            t.prob[Rows*ProbPitch+row*ProbPitch+lane+32]=__float2half_rn(p1-__half2float(high1));
            const float total=sum_warp(p0+p1);
            if(!lane){t.denominator[row]=fmaf(t.denominator[row],adjust,total);t.maximum[row]=next;t.adjust[row]=adjust;}
        }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<Fragment::num_elements;++i){
            const float adjust=t.adjust[group*16+((columns>>(4*i))&15)];
            #pragma unroll
            for(int f=0;f<4;++f)result[f].x[i]*=adjust;
        }
        load_half<Threads>(value,t.kv,first,kh,s.kv_heads,s.valid);__syncthreads();
        #pragma unroll
        for(int offset=0;offset<Keys;offset+=16){
            #pragma unroll
            for(int part=0;part<2;++part){
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
                wmma::load_matrix_sync(b,t.prob+part*Rows*ProbPitch+group*16*ProbPitch+offset,ProbPitch);
                #pragma unroll
                for(int f=0;f<4;++f){
                    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> a;
                    wmma::load_matrix_sync(a,t.kv+offset*Pitch+((warp%4)*4+f)*16,Pitch);
                    wmma::mma_sync(result[f],a,b,result[f]);
                }
            }
        }
        __syncthreads();
        // Bound tensor accumulation depth. History lives on chip in FP32;
        // its softmax basis is updated once per256KV, not once per32KV.
        if((first+Keys)%256==0||first+Keys>=end){
            if(threadIdx.x<Rows){const int r=threadIdx.x;
                t.seam_scale[r]=isfinite(t.seam_maximum[r])?__expf(t.seam_maximum[r]-t.maximum[r]):0.f;
                t.seam_maximum[r]=t.maximum[r];}
            __syncthreads();
            // History shares the exact accumulator type and element mapping.
            // Preserve each scalar's FP32 fma order without a shared round trip.
            #pragma unroll
            for(int i=0;i<Fragment::num_elements;++i){
                const int qrow=group*16+((columns>>(4*i))&15);
                #pragma unroll
                for(int f=0;f<4;++f)
                    history[f].x[i]=fmaf(history[f].x[i],t.seam_scale[qrow],result[f].x[i]);
            }
            #pragma unroll
            for(int f=0;f<4;++f)wmma::fill_fragment(result[f],0.f);
            __syncthreads();
        }
    }
    // Only the final bounded store uses shared memory. Keeping this path
    // handles partial query tiles without an out-of-range WMMA global store.
    auto *temporary=reinterpret_cast<float *>(t.kv);
    #pragma unroll
    for(int g=0;g<Groups;++g){
        if(group==g){
            #pragma unroll
            for(int f=0;f<4;++f)
                wmma::store_matrix_sync(temporary+((warp%4)*4+f)*16,history[f],Dim,wmma::mem_col_major);
        }
        __syncthreads();
        if(group==g){
            #pragma unroll
            for(int row=0;row<16;++row){
                const int qrow=g*16+row,actual=first_query+qrow;
                if(actual<s.tokens){
                    const size_t index=(size_t(actual)*s.query_heads+head)*Dim+local;
                    out[index]=temporary[row*Dim+local]/t.denominator[qrow];
                    out[index+128]=temporary[row*Dim+local+128]/t.denominator[qrow];
                }
            }
        }
        __syncthreads();
    }
}
template<int Groups>void configure(){
    const int bytes=int(sizeof(Keys64QueryTile<Groups>));int device=0;check_cuda(cudaGetDevice(&device),"keys64 online PP setup device");
    if(capabilities(device).shared_per_block<size_t(bytes))throw std::runtime_error("keys64 online PP exceeds shared capacity");
    check_cuda(cudaFuncSetAttribute(keys64_kernel<Groups>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes),"keys64 online PP shared reservation");
    check_cuda(cudaFuncSetAttribute(keys64_kernel<Groups>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"keys64 online PP shared partition");
}
template<int Groups>void launch(const float *q,const __half *k,const __half *v,float *out,PrefillAttentionShape s,int window,cudaStream_t stream){
    keys64_kernel<Groups><<<dim3(s.query_heads,(s.tokens+16*Groups-1)/(16*Groups)),128*Groups,sizeof(Keys64QueryTile<Groups>),stream>>>(q,k,v,out,s,window);
}
__global__ void decode_ring(const unsigned char *k,const unsigned char *v,__half *out,int rows,int origin,int heads,int capacity){
    const size_t count=size_t(rows)*heads*Dim,i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=2*count)return;
    const size_t local=i%count;const int row=int(local/(heads*Dim)),d=int(local%(heads*Dim));
    const auto *source=(i<count?k:v)+(size_t((origin+row)%capacity)*heads*8+d/32)*34;
    out[i]=__float2half_rn(__half2float(*reinterpret_cast<const __half *>(source))*float(reinterpret_cast<const signed char *>(source+2)[d%32]));
}
}
void configure_window_attention(){configure<1>();configure<2>();}
void window_prefill_attention(const float *q,const void *k,const void *v,float *out,void *workspace,size_t bytes,
    PrefillAttentionShape s,int window,cudaStream_t stream){
    if(!q||!k||!v||!out||!workspace||!s.q8||s.dim!=256||s.tokens<1||s.tokens>8192||s.graph_start)
        throw std::invalid_argument("invalid window PP geometry");
    const int origin=std::max(0,s.query_start-window+1),rows=s.valid-origin;
    const size_t plane=size_t(rows)*s.kv_heads*Dim;
    if(rows>s.capacity||bytes<plane*4)throw std::invalid_argument("sliding PP staging exceeds bounded arena");
    auto *half=static_cast<__half *>(workspace);
    decode_ring<<<unsigned((plane*2+255)/256),256,0,stream>>>(static_cast<const unsigned char *>(k),
        static_cast<const unsigned char *>(v),half,rows,origin,s.kv_heads,s.capacity);
    check_cuda(cudaGetLastError(),"bounded sliding PP ring decode");
    s.query_start-=origin;s.valid=rows;s.capacity=rows;
    if(s.tokens<128)launch<1>(q,half,half+plane,out,s,window,stream);
    else launch<2>(q,half,half+plane,out,s,window,stream);
    check_cuda(cudaGetLastError(),"bounded sliding online PP attention");
}
}
