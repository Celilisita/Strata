#include "strata/qwen35/native/attention_score_reuse.hpp"
#include <attention_score_reuse_policy.hpp>
#include <cstdint>
#include <limits>
#include "strata/qwen35/native/ffn.hpp"
#include <mma.h>
#include <cmath>
#include <math_constants.h>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Dim=256,Rows=16,Keys=32,Pitch=272,ProbPitch=48,ScorePitch=40,Partial=258;
constexpr int Group=3;
__device__ float sum_warp(float x){
    for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;
}
__device__ float max_warp(float x){
    for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;
}
__device__ void load_q8(const unsigned char *source,__half *tile,int first,int head,int heads,int valid){
    // Four lanes load the same 34-byte Q8 block; each decodes eight codes.
    // Nothing is expanded to a global FP16 cache.
    for(int i=threadIdx.x;i<Keys*Dim/8;i+=128){
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
__device__ void qk(const __half *q,const __half *k,float *score,int warp){
    if(warp>=2)return;
    wmma::fragment<wmma::accumulator,16,16,16,float> dot;wmma::fill_fragment(dot,0.0f);
    #pragma unroll
    for(int d=0;d<Dim;d+=16){
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
        wmma::load_matrix_sync(a,q+d,Pitch);
        wmma::load_matrix_sync(b,k+warp*16*Pitch+d,Pitch);
        wmma::mma_sync(dot,a,b,dot);
    }
    wmma::store_matrix_sync(score+warp*16,dot,ScorePitch,wmma::mem_row_major);
}
template<int MaximumTokens,bool CacheScores=false>
__global__ void grouped_partial_attention(const float *query,const unsigned char *key,const unsigned char *value,
                                  float *partial,PrefillAttentionShape s,int partitions,float *scores,int cache_keys){
    constexpr int LiveRows=Group*MaximumTokens;
    __shared__ __align__(32) __half q[Rows*Pitch],kv[Keys*Pitch],prob[2*Rows*ProbPitch];
    __shared__ __align__(32) float score[Rows*ScorePitch];
    __shared__ float maximum[Rows],denominator[Rows];
    if constexpr(CacheScores){
        if(s.graph_start)s.query_start=*s.graph_start;
        // Bound the dynamic cursor before arithmetic or any Q/K/V/score read.
        if(s.query_start<0||s.query_start>s.capacity-s.tokens){
            for(int row=0;row<Group*s.tokens;++row){
                const int h=int(blockIdx.x)*Group+row/s.tokens,t=row%s.tokens;
                float *dst=partial+((size_t(h)*partitions+blockIdx.y)*s.tokens+t)*Partial;
                for(int d=threadIdx.x;d<Dim;d+=128)dst[d]=CUDART_NAN_F;
                if(!threadIdx.x){dst[Dim]=0;dst[Dim+1]=1;}
            }
            return;
        }
        s.valid=s.query_start+s.tokens;
    }else if(s.graph_start){s.query_start=*s.graph_start;s.valid=s.query_start+s.tokens;}
    const int head=blockIdx.x*Group,part=blockIdx.y,kh=head/(s.query_heads/s.kv_heads);
    const int live=Group*s.tokens;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int tiles=(s.valid+Keys-1)/Keys,span=(tiles+partitions-1)/partitions;
    const int begin=part*span*Keys,end=min(s.valid,begin+span*Keys);
    float *saved=nullptr;
    if constexpr(CacheScores)saved=scores+(size_t(blockIdx.x)*partitions+part)*LiveRows*cache_keys;
    for(int i=threadIdx.x;i<Rows*Dim;i+=128){
        const int row=i/Dim;
        q[(i/Dim)*Pitch+i%Dim]=row<live?__hmul(__float2half_rn(query[(size_t(row%s.tokens)*s.query_heads+head+row/s.tokens)*Dim+i%Dim]),
                                __float2half_rn(s.scale)):__float2half(0);
    }
    if(threadIdx.x<Rows){maximum[threadIdx.x]=-CUDART_INF_F;denominator[threadIdx.x]=0;}
    __syncthreads();
    // Three query heads sharing one KV head fill live rows of a 16-row tile.
    // The causal token position remains row % s.tokens.
    // Two passes avoid rescaling opaque WMMA accumulator layouts. FP32 tensor
    // partials are periodically merged with ordinary FP32 additions below.
    for(int first=begin;first<end;first+=Keys){
        load_q8(key,kv,first,kh,s.kv_heads,s.valid);__syncthreads();
        qk(q,kv,score,warp);__syncthreads();
        for(int row=warp;row<live;row+=4){
            const int token=first+lane;
            const float x=token<end&&token<=s.query_start+row%s.tokens?score[row*ScorePitch+lane]:-CUDART_INF_F;
            if constexpr(CacheScores)if(token<end)
                saved[size_t(row)*cache_keys+first-begin+lane]=score[row*ScorePitch+lane];
            const float m=max_warp(x);if(!lane)maximum[row]=fmaxf(maximum[row],m);
        }
        __syncthreads();
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> result[4];
    float carry[LiveRows*2]={};
    #pragma unroll
    for(int f=0;f<4;++f)wmma::fill_fragment(result[f],0.0f);
    for(int first=begin;first<end;first+=Keys){
        if constexpr(!CacheScores){
            load_q8(key,kv,first,kh,s.kv_heads,s.valid);__syncthreads();
            qk(q,kv,score,warp);__syncthreads();
        }
        for(int row=warp;row<Rows;row+=4){
            const int token=first+lane;
            const bool active=row<live&&token<end&&token<=s.query_start+row%s.tokens;
            float p;
            if constexpr(CacheScores)p=active?__expf(saved[size_t(row)*cache_keys+first-begin+lane]-maximum[row]):0.0f;
            else p=active?__expf(score[row*ScorePitch+lane]-maximum[row]):0.0f;
            const __half high=__float2half_rn(p);
            prob[row*ProbPitch+lane]=high;
            prob[Rows*ProbPitch+row*ProbPitch+lane]=__float2half_rn(p-__half2float(high));
            const float total=sum_warp(p);if(!lane)denominator[row]+=total;
        }
        __syncthreads();load_q8(value,kv,first,kh,s.kv_heads,s.valid);__syncthreads();
        // High/low probability operands preserve FP32 softmax precision; both
        // multiply the same FP16-decoded Q8 values into one FP32 accumulator.
        #pragma unroll
        for(int key_offset=0;key_offset<Keys;key_offset+=16){
            #pragma unroll
            for(int p=0;p<2;++p){
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
                wmma::load_matrix_sync(b,prob+p*Rows*ProbPitch+key_offset,ProbPitch);
                #pragma unroll
                for(int f=0;f<4;++f){
                    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> a;
                    wmma::load_matrix_sync(a,kv+key_offset*Pitch+(warp*4+f)*16,Pitch);
                    wmma::mma_sync(result[f],a,b,result[f]);
                }
            }
        }
        __syncthreads();
        // Volta tensor accumulators exhibit increasing rounding error over
        // thousands of PV products. Bound their accumulation depth to 256 KV
        // rows, then retain only the live query rows in ordinary FP32 registers.
        if((first-begin)/Keys%8==7||first+Keys>=end){
            auto *tile=reinterpret_cast<float *>(kv);
            #pragma unroll
            for(int f=0;f<4;++f)wmma::store_matrix_sync(tile+(warp*4+f)*16,result[f],Dim,wmma::mem_col_major);
            __syncthreads();
            #pragma unroll
            for(int row=0;row<LiveRows;++row){
                #pragma unroll
                for(int d=0;d<2;++d)if(row<live)carry[row*2+d]+=tile[row*Dim+threadIdx.x+d*128];
            }
            #pragma unroll
            for(int f=0;f<4;++f)wmma::fill_fragment(result[f],0.0f);
            __syncthreads();
        }
    }
    // Scatter into the existing head/partition/token partial layout.
    #pragma unroll
    for(int row=0;row<LiveRows;++row)if(row<live){
        const int h=head+row/s.tokens,t=row%s.tokens;
        float *dst=partial+((size_t(h)*partitions+part)*s.tokens+t)*Partial;
        #pragma unroll
        for(int d=0;d<2;++d)dst[threadIdx.x+d*128]=carry[row*2+d];
        if(!threadIdx.x){dst[Dim]=maximum[row];dst[Dim+1]=denominator[row];}
    }
}

}
VerifyGroupedResources verify_grouped_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,grouped_partial_attention<3>),"grouped verify resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,grouped_partial_attention<3>,128,0),"grouped verify occupancy");
    return {a.numRegs,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
void verify_attention_grouped_q8(const float *q,const void *k,const void *v,float *out,void *scratch,size_t bytes,
                         PrefillAttentionShape s,int partitions,cudaStream_t stream){
    const size_t required=verify_attention_workspace(s,partitions);
    if(!q||!k||!v||!out||!scratch||bytes<required||reinterpret_cast<size_t>(scratch)%16||!s.q8||
       s.kv_heads<1||s.query_heads%s.kv_heads||(s.query_heads/s.kv_heads)%Group||s.query_start<0||s.valid!=s.query_start+s.tokens||
       s.valid>s.capacity||s.capacity>262144||!(s.scale>0)||!std::isfinite(s.scale))
        throw std::invalid_argument("invalid direct Q8 verify invocation");
    if(s.tokens==4)grouped_partial_attention<4><<<dim3(s.query_heads/Group,partitions),128,0,stream>>>(q,
        static_cast<const unsigned char *>(k),static_cast<const unsigned char *>(v),static_cast<float *>(scratch),s,partitions,nullptr,0);
    else grouped_partial_attention<3><<<dim3(s.query_heads/Group,partitions),128,0,stream>>>(q,
        static_cast<const unsigned char *>(k),static_cast<const unsigned char *>(v),static_cast<float *>(scratch),s,partitions,nullptr,0);
    check_cuda(cudaGetLastError(),"own Q8 verify partition");
    verify_attention_merge(static_cast<const float *>(scratch),out,s.query_heads,s.tokens,partitions,stream);
}
VerifyGroupedResources verify_grouped_score_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,grouped_partial_attention<3,true>),"reuse resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,grouped_partial_attention<3,true>,128,0),"reuse occupancy");
    return {a.numRegs,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
VerifyScoreReuseLayout verify_score_reuse_layout(PrefillAttentionShape s,int parts){
    if(s.dim!=256||s.query_heads!=24||s.kv_heads!=4||s.tokens!=3||!s.q8||parts!=32||
       s.capacity<3||s.capacity>16384||s.query_start<0||s.query_start>s.capacity-3||
       s.valid!=s.query_start+3||!(s.scale>0)||!std::isfinite(s.scale))
        throw std::invalid_argument("reuse unsupported shape/capacity");
    const size_t partial=verify_attention_workspace(s,parts),offset=(partial+127)&~size_t(127);
    const int stride=32*((((s.capacity+31)/32)+parts-1)/parts);
    const size_t scores=size_t(8)*parts*9*stride*4;
    return {partial,offset,scores,offset+scores,stride};
}
namespace {
struct Range {uintptr_t first,last;};
Range range(const void *p,size_t bytes,size_t alignment){
    const uintptr_t first=reinterpret_cast<uintptr_t>(p);
    if(!p||first%alignment||bytes>std::numeric_limits<uintptr_t>::max()-first)
        throw std::invalid_argument("reuse pointer/alignment/overflow");
    return {first,first+bytes};
}
bool overlaps(Range a,Range b){return a.first<b.last&&b.first<a.last;}
}
void verify_attention_grouped_score_q8(const float *q,const void *k,const void *v,float *out,void *scratch,size_t bytes,
         PrefillAttentionShape s,int parts,cudaStream_t stream){
    const auto l=verify_score_reuse_layout(s,parts);
    if(bytes<l.total_bytes)throw std::invalid_argument("reuse workspace capacity");
    const size_t vectors=size_t(3)*24*256*4,kv=size_t(s.capacity)*4*272;
    const Range r[]={range(q,vectors,4),range(k,kv,2),range(v,kv,2),range(out,vectors,4),range(scratch,bytes,16)};
    for(int i=0;i<5;++i)for(int j=i+1;j<5;++j)if(overlaps(r[i],r[j]))
        throw std::invalid_argument("reuse live input/output/arena alias");
    if(s.graph_start){
        const auto cursor=range(s.graph_start,4,4);
        for(const auto &x:r)if(overlaps(cursor,x))throw std::invalid_argument("reuse cursor alias");
    }
    auto *scores=reinterpret_cast<float *>(static_cast<unsigned char *>(scratch)+l.score_offset);
    grouped_partial_attention<3,true><<<dim3(8,parts),128,0,stream>>>(q,static_cast<const unsigned char *>(k),
        static_cast<const unsigned char *>(v),static_cast<float *>(scratch),s,parts,scores,l.keys_per_partition);
    check_cuda(cudaGetLastError(),"reuse partition");
    ++attention_score_reuse_request().submissions;
    verify_attention_merge(static_cast<const float *>(scratch),out,24,3,parts,stream);
}
}
