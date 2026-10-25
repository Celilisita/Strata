#include "attention_verify_online.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <mma.h>
#include <cmath>
#include <math_constants.h>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Dim=256,Keys=32,Pitch=Dim+16,ProbPitch=Keys+16,ScorePitch=Keys+8;
template<int Groups>struct OnlineTile {
    static constexpr int Rows=16*Groups;
    __align__(32) __half q[Rows*Pitch],kv[Keys*Pitch],prob[2*Rows*ProbPitch];
    __align__(32) float score[Rows*ScorePitch];
    float maximum[Rows],denominator[Rows],adjust[Rows],seam_maximum[Rows],seam_scale[Rows];
};
__device__ float sum_warp(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float max_warp(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
template<int Threads>__device__ void load_q8(const unsigned char *source,__half *tile,int first,int head,int heads,int valid){
    // Four lanes load the same 34-byte Q8 block; each decodes eight codes.
    // Nothing is expanded to a global FP16 cache.
    for(int i=threadIdx.x;i<Keys*Dim/8;i+=Threads){
        const int token=first+i/32,block=(i%32)/4,offset=(i%4)*8;
        __align__(16) __half values[8];
        if(token<valid){
            const auto *p=source+(size_t(token)*heads+head)*272+block*34;
            const float delta=__half2float(*reinterpret_cast<const __half *>(p));
            const auto *codes=reinterpret_cast<const signed char *>(p+2)+offset;
            #pragma unroll
            for(int j=0;j<8;++j)values[j]=__float2half_rn(delta*float(codes[j]));
        }else{
            #pragma unroll
            for(int j=0;j<8;++j)values[j]=__float2half(0);
        }
        *reinterpret_cast<uint4 *>(tile+(i/32)*Pitch+(i%32)*8)=*reinterpret_cast<const uint4 *>(values);
    }
}
template<int Groups>__device__ void score_tile(OnlineTile<Groups> &t,int warp){
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
template<int Groups>__global__ void verify_online_kernel(const float *query,const unsigned char *key,const unsigned char *value,
                                         float *partial,PrefillAttentionShape s,int partitions){
    constexpr int Rows=16*Groups,Threads=128*Groups,Warps=4*Groups;
    extern __shared__ __align__(32) unsigned char arena[];
    auto &t=*reinterpret_cast<OnlineTile<Groups> *>(arena);
    if(s.graph_start){s.query_start=*s.graph_start;s.valid=s.query_start+s.tokens;}
    const int heads=Groups*3,head=blockIdx.x*heads,part=blockIdx.y,live=heads*s.tokens,kh=head/(s.query_heads/s.kv_heads);
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=threadIdx.x/128,local=threadIdx.x%128;
    const int tiles=(s.valid+Keys-1)/Keys,span=(tiles+partitions-1)/partitions;
    const int begin=part*span*Keys,end=min(s.valid,begin+span*Keys);
    for(int i=threadIdx.x;i<Rows*Dim;i+=Threads){
        const int row=i/Dim;
        t.q[(i/Dim)*Pitch+i%Dim]=row<live?__hmul(__float2half_rn(query[(size_t(row%s.tokens)*s.query_heads+head+row/s.tokens)*Dim+i%Dim]),
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
    for(int first=begin;first<end;first+=Keys){
        load_q8<Threads>(key,t.kv,first,kh,s.kv_heads,s.valid);__syncthreads();
        score_tile(t,warp);__syncthreads();
        for(int row=warp;row<Rows;row+=Warps){
            const int token=first+lane,boundary=s.query_start+row%s.tokens;
            const bool active=row<live&&token<end&&token<=boundary;
            const float score=active?t.score[row*ScorePitch+lane]:-CUDART_INF_F;
            const float next=fmaxf(t.maximum[row],max_warp(score));
            const float adjust=isfinite(t.maximum[row])?__expf(t.maximum[row]-next):0.f;
            const float p=active?__expf(score-next):0.f;
            const __half high=__float2half_rn(p);
            t.prob[row*ProbPitch+lane]=high;
            t.prob[Rows*ProbPitch+row*ProbPitch+lane]=__float2half_rn(p-__half2float(high));
            const float total=sum_warp(p);
            if(!lane){t.denominator[row]=fmaf(t.denominator[row],adjust,total);t.maximum[row]=next;t.adjust[row]=adjust;}
        }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<Fragment::num_elements;++i){
            const float adjust=t.adjust[group*16+((columns>>(4*i))&15)];
            #pragma unroll
            for(int f=0;f<4;++f)result[f].x[i]*=adjust;
        }
        load_q8<Threads>(value,t.kv,first,kh,s.kv_heads,s.valid);__syncthreads();
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
        if((first-begin+Keys)%256==0||first+Keys>=end){
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
                const int qrow=g*16+row;
                if(qrow<live){
                    const int h=head+qrow/s.tokens,token=qrow%s.tokens;
                    float* dst=partial+((size_t(h)*partitions+part)*s.tokens+token)*258;
                    dst[local]=temporary[row*Dim+local];dst[local+128]=temporary[row*Dim+local+128];
                    if(!local){dst[Dim]=t.maximum[qrow];dst[Dim+1]=t.denominator[qrow];}
                }
            }
        }
        __syncthreads();
    }
}
template<int Groups>void configure(){
    const int bytes=int(sizeof(OnlineTile<Groups>));
    check_cuda(cudaFuncSetAttribute(verify_online_kernel<Groups>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes),"onlineQ8 shared reservation");
    check_cuda(cudaFuncSetAttribute(verify_online_kernel<Groups>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"onlineQ8 shared partition");
}
template<int Groups>void launch(const float* q,const void* k,const void* v,float* out,void* workspace,PrefillAttentionShape s,int partitions,cudaStream_t stream){
    verify_online_kernel<Groups><<<dim3(s.query_heads/(Groups*3),partitions),128*Groups,sizeof(OnlineTile<Groups>),stream>>>(q,
        static_cast<const unsigned char*>(k),static_cast<const unsigned char*>(v),static_cast<float*>(workspace),s,partitions);
    check_cuda(cudaGetLastError(),"onlineQ8 verify");
    verify_attention_merge(static_cast<const float*>(workspace),out,s.query_heads,s.tokens,partitions,stream);
}
}
void verify_online_configure(){configure<1>();configure<2>();}
void verify_online_attention(const float* q,const void* k,const void* v,float* out,void* work,size_t bytes,
    PrefillAttentionShape s,int partitions,int group,cudaStream_t stream){
    const auto required=verify_attention_workspace(s,partitions);
    if(!q||!k||!v||!out||!work||bytes<required||!s.q8||s.dim!=256||s.tokens<2||s.tokens>3||
       s.kv_heads<1||s.query_heads%s.kv_heads||(group!=3&&group!=6)||(s.query_heads/s.kv_heads)%group||
       s.query_start<0||s.valid!=s.query_start+s.tokens||s.capacity<s.valid||s.capacity>262144||
       !(s.scale>0)||!std::isfinite(s.scale)||reinterpret_cast<uintptr_t>(work)%16||work==q||work==k||work==v||work==out)
        throw std::invalid_argument("onlineQ8 verify geometry/lease");
    if(group==3)launch<1>(q,k,v,out,work,s,partitions,stream);
    else launch<2>(q,k,v,out,work,s,partitions,stream);
}
}
namespace strata::qwen35::native {
void verify_attention_online_q8(const float* q,const void* k,const void* v,float* out,void* work,size_t bytes,
    PrefillAttentionShape s,int partitions,cudaStream_t stream){
    verify_online_attention(q,k,v,out,work,bytes,s,partitions,6,stream);
}
}
