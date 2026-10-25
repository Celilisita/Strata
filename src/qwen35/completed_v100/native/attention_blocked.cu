#include "strata/qwen35/completed_v100/native/attention_blocked.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>

namespace strata::completed_v100::qwen35::native {
namespace {
struct Layout {
    size_t q,kv,values,scores,bytes;int keys,parts;
    half *query,*key,*value,*hi,*lo;float *score,*result;
};
Layout layout(PrefillAttentionShape s,int rows,void *workspace){
    if(!s.q8||s.graph_start||s.dim!=256||s.tokens<1||s.tokens>8192||s.valid<1||s.valid>262144||
       s.query_heads<1||s.query_heads>256||s.kv_heads<1||s.query_heads%s.kv_heads||
       s.query_start<0||s.query_start+s.tokens!=s.valid||s.capacity<s.valid||s.capacity>262144||
       rows<32||rows>256||rows%32||!(s.scale>0)||!std::isfinite(s.scale))
        throw std::invalid_argument("invalid blocked Q8 attention geometry");
    Layout p{};p.keys=((s.valid+255)/256)*256;p.parts=p.keys/256;
    p.q=size_t(rows)*s.query_heads*s.dim;p.kv=size_t(p.keys)*s.kv_heads*s.dim;
    p.values=size_t(p.keys)*s.query_heads*s.dim;p.scores=size_t(rows)*s.query_heads*p.keys;
    // Softmax finishes reading scores before PV. With dim/partition256,
    // partial output has exactly the same extent as the now-dead score arena.
    p.bytes=2*p.q+2*p.kv+2*p.values+8*p.scores;
    if(workspace){auto *b=static_cast<unsigned char *>(workspace);
        p.query=reinterpret_cast<half *>(b);b+=2*p.q;
        p.key=reinterpret_cast<half *>(b);b+=2*p.kv;p.value=reinterpret_cast<half *>(b);b+=2*p.values;
        p.score=reinterpret_cast<float *>(b);b+=4*p.scores;
        p.hi=reinterpret_cast<half *>(b);b+=2*p.scores;p.lo=reinterpret_cast<half *>(b);b+=2*p.scores;
        p.result=p.score;
    }
    return p;
}
__global__ void pack_kv(const unsigned char *k,const unsigned char *v,half *kh,half *vh,
                         PrefillAttentionShape s,int keys){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(keys)*s.query_heads*s.dim)return;
    const int d=i%s.dim,t=(i/s.dim)%keys,h=i/(size_t(s.dim)*keys);
    half kd=__float2half(0),vd=kd;
    if(t<s.valid){const size_t at=(size_t(t)*s.kv_heads+h/(s.query_heads/s.kv_heads))*272+(d/32)*34;
        kd=__float2half_rn(__half2float(*reinterpret_cast<const half *>(k+at))*float(reinterpret_cast<const signed char *>(k+at+2)[d%32]));
        vd=__float2half_rn(__half2float(*reinterpret_cast<const half *>(v+at))*float(reinterpret_cast<const signed char *>(v+at+2)[d%32]));
    }
    if(h%(s.query_heads/s.kv_heads)==0)kh[(size_t(h/(s.query_heads/s.kv_heads))*keys+t)*s.dim+d]=kd;
    vh[i]=vd;
}
__global__ void pack_query(const float *q,half *packed,PrefillAttentionShape s,int first,int rows){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*s.query_heads*s.dim)return;
    const int d=i%s.dim,t=(i/s.dim)%rows,h=i/(size_t(rows)*s.dim);
    packed[i]=__hmul(__float2half_rn(q[(size_t(first+t)*s.query_heads+h)*s.dim+d]),__float2half_rn(s.scale));
}
__device__ float warp_max(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float warp_sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__global__ void softmax(const float *score,half *hi,half *lo,PrefillAttentionShape s,int first,int rows,int keys){
    __shared__ float reductions[8];const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int last=s.query_start+first+int(blockIdx.x)%rows;
    const size_t start=size_t(blockIdx.x)*keys;float maximum=-CUDART_INF_F;
    for(int k=tid;k<keys;k+=256)if(k<=last)maximum=fmaxf(maximum,score[start+k]);
    maximum=warp_max(maximum);if(!lane)reductions[warp]=maximum;__syncthreads();
    maximum=warp_max(lane<8?reductions[lane]:-CUDART_INF_F);__syncthreads();float sum=0;
    for(int k=tid;k<keys;k+=256)if(k<=last)sum+=__expf(score[start+k]-maximum);
    sum=warp_sum(sum);if(!lane)reductions[warp]=sum;__syncthreads();
    sum=warp_sum(lane<8?reductions[lane]:0.f);
    for(int k=tid;k<keys;k+=256){
        const float p=k<=last?__expf(score[start+k]-maximum)/sum:0.f;
        const int head=int(blockIdx.x)/rows,row=int(blockIdx.x)%rows;
        const size_t at=((size_t(head)*(keys/256)+k/256)*rows+row)*256+k%256;
        const half high=__float2half_rn(p);hi[at]=high;
        lo[at]=__float2half_rn((p-__half2float(high))*4096.f);
    }
}
__global__ void scatter(const float *result,float *out,PrefillAttentionShape s,int first,int rows,int parts){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*s.query_heads*s.dim)return;
    const int d=i%s.dim,h=(i/s.dim)%s.query_heads,t=i/(size_t(s.dim)*s.query_heads);
    float sum=0;
    for(int p=0;p<parts;++p)sum+=result[((size_t(h)*parts+p)*rows+t)*s.dim+d];
    out[(size_t(first+t)*s.query_heads+h)*s.dim+d]=sum;
}
}
size_t blocked_attention_workspace(PrefillAttentionShape s,int rows){return layout(s,rows,nullptr).bytes;}
void blocked_attention(DenseGemm &gemm,const float *q,const void *k,const void *v,float *out,
                       void *workspace,size_t bytes,PrefillAttentionShape s,int tile,cudaStream_t stream){
    auto p=layout(s,tile,workspace);
    if(!q||!k||!v||!out||!workspace||bytes<p.bytes||reinterpret_cast<size_t>(workspace)%256)
        throw std::invalid_argument("blocked Q8 attention arena lease");
    pack_kv<<<unsigned((p.values+255)/256),256,0,stream>>>(static_cast<const unsigned char *>(k),
        static_cast<const unsigned char *>(v),p.key,p.value,s,p.keys);
    const int group=s.query_heads/s.kv_heads;const size_t kv_head=size_t(p.keys)*s.dim;
    for(int first=0;first<s.tokens;first+=tile){
        const int rows=(std::min)(tile,s.tokens-first);const size_t q_head=size_t(rows)*s.dim,score_head=size_t(rows)*p.keys;
        pack_query<<<unsigned((size_t(rows)*s.query_heads*s.dim+255)/256),256,0,stream>>>(q,p.query,s,first,rows);
        GemmBatch qk{p.keys,rows,s.dim,s.dim,s.dim,p.keys,group,0,(long long)q_head,(long long)score_head,true,false};
        for(int h=0;h<s.kv_heads;++h)gemm.batched_fp16(p.key+h*kv_head,p.query+h*group*q_head,
            p.score+h*group*score_head,qk,stream);
        softmax<<<rows*s.query_heads,256,0,stream>>>(p.score,p.hi,p.lo,s,first,rows,p.keys);
        // Each batched dot has exactly256 KV operands. V is head-replicated
        // once so every head/partition can share one strided-batched dispatch.
        const long long partial_head=(long long)rows*s.dim;
        GemmBatch pv{s.dim,rows,256,s.dim,256,s.dim,s.query_heads*p.parts,
            256LL*s.dim,rows*256LL,partial_head};
        gemm.batched_fp16(p.value,p.hi,p.result,pv,stream);
        pv.alpha=1.f/4096;pv.beta=1.f;
        gemm.batched_fp16(p.value,p.lo,p.result,pv,stream);
        scatter<<<unsigned((size_t(rows)*s.query_heads*s.dim+255)/256),256,0,stream>>>(p.result,out,s,first,rows,p.parts);
    }
    check_cuda(cudaGetLastError(),"blocked Q8 attention pack/softmax/scatter");
}
}
