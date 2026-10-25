#include "pp_gqa.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>

namespace strata::qwen35::native::pp_gqa {
namespace n=strata::qwen35::native;
using n::PrefillAttentionShape; using n::DenseGemm; using n::GemmBatch; using n::check_cuda;
namespace {
struct Layout {
    size_t q,kv,values,scores,bytes;int keys,parts;
    half *query,*key,*value,*hi,*lo;float *score,*result;
};
Layout layout(PrefillAttentionShape s,int rows,void *workspace,int part){
    if((part!=256&&part!=512)||s.valid>32768||!s.q8||s.graph_start||s.dim!=256||s.tokens<1||s.tokens>8192||s.valid<1||s.valid>262144||
       s.query_heads<1||s.query_heads>256||s.kv_heads<1||s.query_heads%s.kv_heads||
       s.query_start<0||s.query_start+s.tokens!=s.valid||s.capacity<s.valid||s.capacity>262144||
       rows<32||rows>256||rows%32||!(s.scale>0)||!std::isfinite(s.scale))
        throw std::invalid_argument("invalid blocked Q8 attention geometry");
    Layout p{};p.keys=((s.valid+part-1)/part)*part;p.parts=p.keys/part;
    p.q=size_t(rows)*s.query_heads*s.dim;p.kv=size_t(p.keys)*s.kv_heads*s.dim;
    p.values=p.kv;p.scores=size_t(rows)*s.query_heads*p.keys;
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
    if(i>=size_t(keys)*s.kv_heads*s.dim)return;
    const int d=i%s.dim,t=(i/s.dim)%keys,h=i/(size_t(s.dim)*keys);
    half kd=__float2half(0),vd=kd;
    if(t<s.valid){const size_t at=(size_t(t)*s.kv_heads+h)*272+(d/32)*34;
        kd=__float2half_rn(__half2float(*reinterpret_cast<const half *>(k+at))*float(reinterpret_cast<const signed char *>(k+at+2)[d%32]));
        vd=__float2half_rn(__half2float(*reinterpret_cast<const half *>(v+at))*float(reinterpret_cast<const signed char *>(v+at+2)[d%32]));
    }
    kh[i]=kd;
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
template<int Part>
__global__ __launch_bounds__(1024) void softmax_cached(const float* score,half* hi,half* lo,
    PrefillAttentionShape s,int first,int rows,int keys){
    constexpr int Threads=1024,Items=32;
    __shared__ float reductions[32];
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int last=s.query_start+first+int(blockIdx.x)%rows;
    const size_t start=size_t(blockIdx.x)*keys;
    float values[Items],maximum=-CUDART_INF_F;
    #pragma unroll
    for(int i=0;i<Items;++i){
        const int k=tid+i*Threads;
        const float v=k<keys&&k<=last?score[start+k]:-CUDART_INF_F;
        values[i]=v;maximum=fmaxf(maximum,v);
    }
    maximum=warp_max(maximum);
    if(!lane)reductions[warp]=maximum;
    __syncthreads();
    maximum=warp_max(lane<32?reductions[lane]:-CUDART_INF_F);
    __syncthreads();float sum=0;
    #pragma unroll
    for(int i=0;i<Items;++i){values[i]=__expf(values[i]-maximum);sum+=values[i];}
    sum=warp_sum(sum);if(!lane)reductions[warp]=sum;__syncthreads();
    sum=warp_sum(lane<32?reductions[lane]:0.f);
    const int head=int(blockIdx.x)/rows,row=int(blockIdx.x)%rows,group=s.query_heads/s.kv_heads;
    #pragma unroll
    for(int i=0;i<Items;++i){
        const int k=tid+i*Threads;
        if(k<keys){
            const float p=values[i]/sum;
            const size_t at=((size_t(head/group)*(keys/Part)+k/Part)*(group*rows)
                             +(head%group)*rows+row)*Part+k%Part;
            const half high=__float2half_rn(p);hi[at]=high;
            lo[at]=__float2half_rn((p-__half2float(high))*4096.f);
        }
    }
}
__global__ void scatter(const float *result,float *out,PrefillAttentionShape s,int first,int rows,int parts){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*s.query_heads*s.dim)return;
    const int d=i%s.dim,h=(i/s.dim)%s.query_heads,t=i/(size_t(s.dim)*s.query_heads);
    float sum=0;
    const int group=s.query_heads/s.kv_heads;
    for(int p=0;p<parts;++p)sum+=result[((size_t(h/group)*parts+p)*(group*rows)
                                     +(h%group)*rows+t)*s.dim+d];
    out[(size_t(first+t)*s.query_heads+h)*s.dim+d]=sum;
}
}
Resources resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,softmax_cached<512>),"PP GQA resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,softmax_cached<512>,1024,0),"PP GQA occupancy");
    return {a.numRegs,blocks,int(a.localSizeBytes),int(a.sharedSizeBytes)};
}
size_t workspace(PrefillAttentionShape s,int rows){return layout(s,rows,nullptr,512).bytes;}
void run(DenseGemm &gemm,const float *q,const void *k,const void *v,float *out,
                       void *workspace,size_t bytes,PrefillAttentionShape s,int tile,cudaStream_t stream){
    constexpr int part=512;
    auto p=layout(s,tile,workspace,part);
    if(!q||!k||!v||!out||!workspace||bytes<p.bytes||reinterpret_cast<size_t>(workspace)%256)
        throw std::invalid_argument("blocked Q8 attention arena lease");
    pack_kv<<<unsigned((p.values+255)/256),256,0,stream>>>(static_cast<const unsigned char *>(k),
        static_cast<const unsigned char *>(v),p.key,p.value,s,p.keys);
    const int group=s.query_heads/s.kv_heads;const size_t kv_head=size_t(p.keys)*s.dim;
    for(int first=0;first<s.tokens;first+=tile){
        const int rows=(std::min)(tile,s.tokens-first);const size_t q_head=size_t(rows)*s.dim,score_head=size_t(rows)*p.keys;
        pack_query<<<unsigned((size_t(rows)*s.query_heads*s.dim+255)/256),256,0,stream>>>(q,p.query,s,first,rows);
        // Heads sharing one KV head form adjacent columns in the same GEMM.
        // This preserves the original head/row/key score memory layout.
        GemmBatch qk{p.keys,group*rows,s.dim,s.dim,s.dim,p.keys,s.kv_heads,
            (long long)kv_head,group*(long long)q_head,group*(long long)score_head,true,false};
        gemm.batched_fp16(p.key,p.query,p.score,qk,stream);
        if(part==256)softmax_cached<256><<<rows*s.query_heads,1024,0,stream>>>(p.score,p.hi,p.lo,s,first,rows,p.keys);
        else softmax_cached<512><<<rows*s.query_heads,1024,0,stream>>>(p.score,p.hi,p.lo,s,first,rows,p.keys);
        // Preserve exactly256 KV operands per F32 partial; grouped heads reuse one V payload.
        const long long partial_head=(long long)group*rows*s.dim;
        GemmBatch pv{s.dim,group*rows,part,s.dim,part,s.dim,s.kv_heads*p.parts,
            (long long)part*s.dim,(long long)group*rows*part,partial_head};
        gemm.batched_fp16(p.value,p.hi,p.result,pv,stream);
        pv.alpha=1.f/4096;pv.beta=1.f;
        gemm.batched_fp16(p.value,p.lo,p.result,pv,stream);
        scatter<<<unsigned((size_t(rows)*s.query_heads*s.dim+255)/256),256,0,stream>>>(p.result,out,s,first,rows,p.parts);
    }
    check_cuda(cudaGetLastError(),"blocked Q8 attention pack/softmax/scatter");
}
}
