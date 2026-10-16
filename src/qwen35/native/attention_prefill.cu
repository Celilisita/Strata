#include "strata/qwen35/native/attention_prefill.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "attention_value.cuh"
#include <cmath>
#include <cfloat>
#include <math_constants.h>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Queries=16,Keys=32,Dimension=256;
__device__ float reduce_sum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float reduce_max(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ __half cache_half(const unsigned char *cache,int token,int head,int d,int heads,int valid,bool q8){
    if(token>=valid)return __float2half(0);
    const auto *p=cache+(size_t(token)*heads+head)*(q8?272:144)+(d/32)*(q8?34:18);
    if(q8)return __float2half_rn(__fmul_rn(__half2float(*reinterpret_cast<const __half *>(p)),
                                        float(reinterpret_cast<const signed char *>(p+2)[d%32])));
    const int code=(p[2+d%16]>>(d%32<16?0:4))&15;
    return __float2half_rn(__fmul_rn(__half2float(*reinterpret_cast<const __half *>(p)),float(code-8)));
}
__global__ void decode_cache(const unsigned char *k,const unsigned char *v,__half *kh,__half *vh,int heads,int valid,bool q8,const int *start,int tokens){
    if(start)valid=*start+tokens;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(valid)*heads*Dimension;i+=size_t(gridDim.x)*blockDim.x){
        const int d=i%Dimension,head=(i/Dimension)%heads,token=i/(Dimension*heads);
        kh[i]=cache_half(k,token,head,d,heads,valid,q8);vh[i]=cache_half(v,token,head,d,heads,valid,q8);
    }
}
__device__ void load_cache(const __half *input,__half *tile,int first,int head,int heads,int valid){
    for(int i=threadIdx.x;i<Keys*Dimension/8;i+=128){
        const int token=first+i/(Dimension/8),d=(i%(Dimension/8))*8;
        uint4 x{};
        if(token<valid)x=*reinterpret_cast<const uint4 *>(input+(size_t(token)*heads+head)*Dimension+d);
        *reinterpret_cast<uint4 *>(tile+i*8)=x;
    }
}
__device__ int segment(int tile,int step,int steps,int work,int segments){
    return int(((static_cast<long long>(tile)*steps+step+1)*segments-1)/work);
}
__global__ void attention(const float *query,const __half *key,const __half *value,float *out,
                          PrefillAttentionShape s,PrefillRounding r){
    __shared__ __align__(32) __half q[Queries*Dimension],kv[Keys*Dimension],prob[2*Queries*Keys];
    __shared__ __align__(32) float score[Queries*Keys],carry[Queries*Dimension];
    __shared__ float maximum[2*Queries],denominator[4*Queries],rescale[2*Queries];
    __shared__ float combined_max[Queries],combined_sum[Queries],old_scale[Queries],new_scale[Queries];
    if(s.graph_start){s.query_start=*s.graph_start;s.valid=s.query_start+s.tokens;r.padded_keys=((s.valid+255)/256)*256;}
    const int head=blockIdx.x,first_query=blockIdx.y*Queries,kh=head/(s.query_heads/s.kv_heads);
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,gqa=s.query_heads/s.kv_heads;
    const int qgroups=(s.tokens+r.query_group-1)/r.query_group,hgroups=(gqa+r.head_group-1)/r.head_group;
    const int tile=(kh*hgroups+(head%gqa)/r.head_group)*qgroups+first_query/r.query_group;
    const int steps=r.padded_keys/Keys,work=s.kv_heads*hgroups*qgroups*steps;
    for(int i=threadIdx.x;i<Queries*Dimension;i+=128){
        const int t=first_query+i/Dimension,d=i%Dimension;carry[i]=0;
        q[i]=__hmul(__float2half_rn(t<s.tokens?query[(size_t(t)*s.query_heads+head)*Dimension+d]:0.0f),__float2half_rn(s.scale));
    }
    if(threadIdx.x<2*Queries)maximum[threadIdx.x]=-FLT_MAX/2;
    if(threadIdx.x<4*Queries)denominator[threadIdx.x]=0;
    if(threadIdx.x<Queries){combined_max[threadIdx.x]=-FLT_MAX/2;combined_sum[threadIdx.x]=0;}
    PrefillValues result;result.clear();__syncthreads();
    const int last_key=min(s.valid,s.query_start+min(s.tokens,first_query+Queries));
    for(int first=0;first<last_key;first+=Keys){
        load_cache(key,kv,first,kh,s.kv_heads,s.valid);
        __syncthreads();
        if(warp<2){
            wmma::fragment<wmma::accumulator,16,16,16,float> dots;wmma::fill_fragment(dots,0.0f);
            #pragma unroll
            for(int d=0;d<Dimension;d+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
                wmma::load_matrix_sync(a,q+d,Dimension);wmma::load_matrix_sync(b,kv+warp*16*Dimension+d,Dimension);
                wmma::mma_sync(dots,a,b,dots);
            }
            wmma::store_matrix_sync(score+warp*16,dots,Keys,wmma::mem_row_major);
        }
        __syncthreads();
        for(int row=warp;row<Queries;row+=4){
            const bool masked=first+lane>=s.valid || first+lane>s.query_start+first_query+row || first_query+row>=s.tokens;
            const float x=masked?-CUDART_INF_F:score[row*Keys+lane];
            #pragma unroll
            for(int part=0;part<2;++part){
                const bool active=(lane/8)%2==part;
                const float m=reduce_max(active?x:-FLT_MAX/2);
                const int index=part*Queries+row;
                const float next=fmaxf(maximum[index],__fadd_rn(m,__fmul_rn(3.0f,.6931f)));
                const float old_diff=__fsub_rn(maximum[index],next),adjust=old_diff>=-20?__expf(old_diff):0.0f;
                // Small probabilities still participate in the F32 denominator even
                // when their half conversion is zero. Only history rescaling flushes.
                const float diff=__fsub_rn(x,next),p=active?__expf(diff):0.0f;
                prob[part*Queries*Keys+row*Keys+lane]=__float2half_rn(p);
                // Two F32 sub-sums survive until this numeric interval ends.
                // The scalar sequence is part of the half-operand model contract.
                const int slot=lane&1;float sum=0;
                #pragma unroll
                for(int group=0;group<2;++group){
                    const int base=part*8+group*16+slot*2;
                    sum=__fadd_rn(sum,__shfl_sync(0xffffffff,p,base));
                    sum=__fadd_rn(sum,__shfl_sync(0xffffffff,p,base+1));
                    sum=__fadd_rn(sum,__shfl_sync(0xffffffff,p,base+4));
                    sum=__fadd_rn(sum,__shfl_sync(0xffffffff,p,base+5));
                }
                if(lane<2)denominator[index*2+slot]=__fmaf_rn(denominator[index*2+slot],adjust,sum);
                if(!lane){maximum[index]=next;rescale[index]=adjust;}
            }
        }
        __syncthreads();result.rescale(rescale,kv,warp,threadIdx.x);
        load_cache(value,kv,first,kh,s.kv_heads,s.valid);
        __syncthreads();result.accumulate(kv,prob,warp,lane);__syncthreads();
        const int step=first/Keys;
        const bool seam=first+Keys>=last_key || segment(tile,step,steps,work,r.segments)!=segment(tile,step+1,steps,work,r.segments);
        if(seam){
            result.store(kv,warp,lane);__syncthreads();
            if(threadIdx.x<Queries){
                const int row=threadIdx.x;
                const float partial_max=fmaxf(maximum[row],maximum[Queries+row]);
                const float p0=__expf(__fsub_rn(maximum[row],partial_max)),p1=__expf(__fsub_rn(maximum[Queries+row],partial_max));
                const float d0=__fadd_rn(denominator[row*2],denominator[row*2+1]);
                const float d1=__fadd_rn(denominator[(Queries+row)*2],denominator[(Queries+row)*2+1]);
                // Stripe 1 product rounds before stripe 0 is fused into it.
                const float partial_sum=__fmaf_rn(p0,d0,__fmul_rn(p1,d1));
                const float next=fmaxf(combined_max[row],partial_max);
                const float old_diff=__fsub_rn(combined_max[row],next),diff=__fsub_rn(partial_max,next);
                const float old=old_diff>=-20?__expf(old_diff):0.0f,fresh=diff>=-20?__expf(diff):0.0f;
                old_scale[row]=old;new_scale[row]=fresh;rescale[row]=p0;rescale[Queries+row]=p1;
                combined_sum[row]=__fmaf_rn(fresh,partial_sum,__fmul_rn(combined_sum[row],old));
                combined_max[row]=next;
            }
            __syncthreads();
            for(int i=threadIdx.x;i<Queries*Dimension;i+=128){
                const int row=i/Dimension;
                const float partial=__fmaf_rn(__half2float(kv[Queries*Dimension+i]),rescale[Queries+row],
                    __fmul_rn(__half2float(kv[i]),rescale[row]));
                carry[i]=__fmaf_rn(partial,new_scale[row],__fmul_rn(carry[i],old_scale[row]));
            }
            if(threadIdx.x<2*Queries)maximum[threadIdx.x]=-FLT_MAX/2;
            if(threadIdx.x<4*Queries)denominator[threadIdx.x]=0;
            result.clear();__syncthreads();
        }
    }
    for(int i=threadIdx.x;i<Queries*Dimension;i+=128){
        const int row=i/Dimension,t=first_query+row;
        if(t<s.tokens)out[(size_t(t)*s.query_heads+head)*Dimension+i%Dimension]=__fdividef(carry[i],combined_sum[row]);
    }
}
}
size_t prefill_attention_workspace(PrefillAttentionShape s){
    if(s.dim!=Dimension || s.kv_heads<1 || s.kv_heads>256 || s.valid<1 || s.valid>262144)
        throw std::invalid_argument("invalid own PP cache staging extent");
    return size_t(s.graph_start?s.capacity:s.valid)*s.kv_heads*s.dim*2*sizeof(__half);
}
void prefill_decode_kv(const void *k,const void *v,void *workspace,size_t workspace_bytes,
                       PrefillAttentionShape s,cudaStream_t stream){
    if(!k||!v||s.dim!=Dimension||s.tokens<1||s.tokens>8192||s.query_start<0||
       s.query_start+s.tokens!=s.valid||s.valid>s.capacity||s.capacity>262144||
       (s.graph_start&&(s.rounding.segments<1||s.rounding.segments>4096)))
        throw std::invalid_argument("invalid borrowed PP KV decode invocation");
    const size_t required=prefill_attention_workspace(s);
    if(!workspace || workspace_bytes<required || reinterpret_cast<size_t>(workspace)%16)
        throw std::invalid_argument("invalid own borrowed PP cache workspace");
    auto *half_key=static_cast<__half *>(workspace),*half_value=half_key+required/(2*sizeof(__half));
    const size_t count=size_t(s.valid)*s.kv_heads*s.dim;
    decode_cache<<<s.graph_start?unsigned(s.rounding.segments*4):unsigned((count+255)/256),256,0,stream>>>(static_cast<const unsigned char *>(k),
        static_cast<const unsigned char *>(v),half_key,half_value,s.kv_heads,s.valid,s.q8,s.graph_start,s.tokens);
    check_cuda(cudaGetLastError(),"own once-per-invocation KV decode");
 }
void prefill_attention(const float *q,const void *k,const void *v,float *out,void *workspace,size_t workspace_bytes,
                       PrefillAttentionShape s,cudaStream_t stream){
    const auto r=s.rounding;
    if(!q || !k || !v || !out || s.dim!=256 || s.query_heads<1 || s.query_heads>256 || s.kv_heads<1 || s.query_heads%s.kv_heads ||
       s.tokens<1 || s.tokens>8192 || s.query_start<0 || s.valid<s.tokens || s.query_start>s.valid-s.tokens ||
       s.valid>s.capacity || s.capacity>262144 || !(s.scale>0) || !std::isfinite(s.scale) || r.query_group!=32 || r.head_group!=2 ||
       r.segments<1 || r.padded_keys<s.valid || r.padded_keys>262144 || r.padded_keys%32)
        throw std::invalid_argument("invalid own PP attention geometry or explicit rounding contract");
    prefill_decode_kv(k,v,workspace,workspace_bytes,s,stream);
    const size_t required=prefill_attention_workspace(s);
    auto *half_key=static_cast<__half *>(workspace),*half_value=half_key+required/(2*sizeof(__half));
    attention<<<dim3(s.query_heads,(s.tokens+Queries-1)/Queries),128,0,stream>>>(q,half_key,half_value,out,s,r);
    check_cuda(cudaGetLastError(),"own bounded independent PP attention");
}
}
