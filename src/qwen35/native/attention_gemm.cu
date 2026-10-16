#include "strata/qwen35/native/attention_gemm.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>

namespace strata::qwen35::native {
namespace {
__global__ void pack_query(const float *q,half *out,PrefillAttentionShape s){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(s.tokens)*s.query_heads*s.dim)return;
    const int d=i%s.dim,t=(i/s.dim)%s.tokens,h=i/(size_t(s.dim)*s.tokens);
    out[i]=__hmul(__float2half_rn(q[(size_t(t)*s.query_heads+h)*s.dim+d]),__float2half_rn(s.scale));
}
__device__ half cached(const unsigned char *p,int d){
    p+=(d/32)*18;const int code=(p[2+d%16]>>(d%32<16?0:4))&15;
    return __float2half_rn(__fmul_rn(__half2float(*reinterpret_cast<const half *>(p)),float(code-8)));
}
__global__ void pack_kv(const unsigned char *k,const unsigned char *v,half *kh,half *vh,PrefillAttentionShape s){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(s.valid)*s.kv_heads*s.dim)return;
    const int d=i%s.dim,t=(i/s.dim)%s.valid,h=i/(size_t(s.dim)*s.valid);
    const size_t at=(size_t(t)*s.kv_heads+h)*(s.dim/32)*18;
    kh[i]=cached(k+at,d);vh[i]=cached(v+at,d);
}
__device__ float maximum(float v){for(int d=16;d;d/=2)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));return v;}
__device__ float sum(float v){for(int d=16;d;d/=2)v+=__shfl_xor_sync(0xffffffff,v,d);return v;}
__global__ void probabilities(const float *scores,half *hi,half *lo,PrefillAttentionShape s){
    __shared__ float reductions[8];const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int row=blockIdx.x,query=row%s.tokens,last=s.query_start+query;
    const size_t start=size_t(row)*s.valid;float m=-CUDART_INF_F;
    for(int k=tid;k<s.valid;k+=256)if(k<=last)m=fmaxf(m,scores[start+k]);
    m=maximum(m);if(!lane)reductions[warp]=m;__syncthreads();
    m=maximum(lane<8?reductions[lane]:-CUDART_INF_F);__syncthreads();float denominator=0;
    for(int k=tid;k<s.valid;k+=256)if(k<=last)denominator+=__expf(scores[start+k]-m);
    denominator=sum(denominator);if(!lane)reductions[warp]=denominator;__syncthreads();
    denominator=sum(lane<8?reductions[lane]:0.0f);
    for(int k=tid;k<s.valid;k+=256){
        const float p=k<=last?__expf(scores[start+k]-m)/denominator:0.0f;
        const half rounded=__float2half_rn(p);hi[start+k]=rounded;
        // Power-of-two scaling protects the low residual from half underflow.
        lo[start+k]=__float2half_rn((p-__half2float(rounded))*4096.0f);
    }
}
__global__ void scatter(const float *head_output,float *out,PrefillAttentionShape s){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(s.tokens)*s.query_heads*s.dim)return;
    const int d=i%s.dim,h=(i/s.dim)%s.query_heads,t=i/(size_t(s.dim)*s.query_heads);
    out[i]=head_output[(size_t(h)*s.tokens+t)*s.dim+d];
}
}
AttentionGemmLayout attention_gemm_layout(PrefillAttentionShape s,void *workspace){
    if(s.dim!=256||s.tokens<1||s.tokens>8192||s.valid<1||s.valid>262144||s.query_heads<1||s.kv_heads<1||
       s.query_heads%s.kv_heads||s.query_start<0||s.query_start+s.tokens!=s.valid)
        throw std::invalid_argument("invalid causal attention GEMM geometry");
    AttentionGemmLayout p{size_t(s.tokens)*s.query_heads*s.dim,size_t(s.valid)*s.kv_heads*s.dim,
                          size_t(s.tokens)*s.query_heads*s.valid,0};
    p.bytes=2*(p.q_elements+2*p.kv_elements)+8*p.score_elements;
    if(workspace){auto *b=static_cast<unsigned char *>(workspace);
        p.query=b;b+=2*p.q_elements;p.key=b;b+=2*p.kv_elements;p.value=b;b+=2*p.kv_elements;
        p.scores=b;b+=4*p.score_elements;p.probability_hi=b;b+=2*p.score_elements;p.probability_lo=b;
    }
    return p;
}
bool gemm_prefill_attention(DenseGemm &gemm,const float *q,const void *k,const void *v,float *out,float *temporary,
                            void *workspace,size_t capacity,PrefillAttentionShape s,cudaStream_t stream){
    auto p=attention_gemm_layout(s,workspace);
    if(s.graph_start||s.q8||s.tokens<128||p.bytes>capacity)return false;
    if(!q||!k||!v||!out||!temporary||!workspace||out==temporary)throw std::invalid_argument("attention GEMM workspace lease invalid");
    pack_query<<<unsigned((p.q_elements+255)/256),256,0,stream>>>(q,static_cast<half *>(p.query),s);
    pack_kv<<<unsigned((p.kv_elements+255)/256),256,0,stream>>>(static_cast<const unsigned char *>(k),static_cast<const unsigned char *>(v),static_cast<half *>(p.key),static_cast<half *>(p.value),s);
    check_cuda(cudaGetLastError(),"attention GEMM operand packing");
    const int group=s.query_heads/s.kv_heads;const size_t q_head=size_t(s.tokens)*s.dim,kv_head=size_t(s.valid)*s.dim,score_head=size_t(s.tokens)*s.valid;
    auto *qh=static_cast<half *>(p.query),*kh=static_cast<half *>(p.key),*vh=static_cast<half *>(p.value);
    auto *score=static_cast<float *>(p.scores);auto *hi=static_cast<half *>(p.probability_hi),*lo=static_cast<half *>(p.probability_lo);
    GemmBatch qk{s.valid,s.tokens,s.dim,s.dim,s.dim,s.valid,group,0,(long long)q_head,(long long)score_head,true,false};
    for(int h=0;h<s.kv_heads;++h)gemm.batched_fp16(kh+h*kv_head,qh+h*group*q_head,score+h*group*score_head,qk,stream);
    probabilities<<<s.tokens*s.query_heads,256,0,stream>>>(score,hi,lo,s);
    check_cuda(cudaGetLastError(),"attention stable softmax and probability decomposition");
    GemmBatch pv{s.dim,s.tokens,s.valid,s.dim,s.valid,s.dim,group,0,(long long)score_head,(long long)q_head};
    for(int h=0;h<s.kv_heads;++h){
        // Bound each FP32 dot's reduction depth, then combine in the FP32 output.
        // No extra storage; the original probability and KV operands stay live.
        for(int begin=0;begin<s.valid;begin+=256){
            pv.k=(std::min)(256,s.valid-begin);
            pv.alpha=1.0f;pv.beta=begin?1.0f:0.0f;
            gemm.batched_fp16(vh+h*kv_head+size_t(begin)*s.dim,hi+h*group*score_head+begin,
                              temporary+h*group*q_head,pv,stream);
            pv.alpha=1.0f/4096;pv.beta=1.0f;
            gemm.batched_fp16(vh+h*kv_head+size_t(begin)*s.dim,lo+h*group*score_head+begin,
                              temporary+h*group*q_head,pv,stream);
        }
    }
    scatter<<<unsigned((p.q_elements+255)/256),256,0,stream>>>(temporary,out,s);
    check_cuda(cudaGetLastError(),"attention GEMM output layout");return true;
}
}
