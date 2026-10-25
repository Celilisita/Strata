#include "rescale_skip_attention.hpp"
#include "pipeline_attention.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <mma.h>
#include <cmath>
#include <math_constants.h>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Dim=256,Keys=64,Pitch=Dim+16,ScorePitch=Keys+8,ProbPitch=2*ScorePitch;
union __align__(32) ScoreProbability {float score[ScorePitch];__half probability[ProbPitch];};
static_assert(sizeof(ScoreProbability)==ScorePitch*sizeof(float),"row alias extent");
static_assert((ProbPitch*sizeof(__half))%32==0,"probability row WMMA alignment");
template<int Groups>struct CompactQueryTile {
    static constexpr int Rows=16*Groups;
    __align__(32) __half q[Rows*Pitch],kv[Keys*Pitch];
    ScoreProbability row[Rows];
    float maximum[Rows],denominator[Rows],adjust[Rows],seam_maximum[Rows],seam_scale[Rows];
};
__device__ float sum_warp(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float max_warp(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
// Scores are dead per row after the warp has read all64 into registers.
// One RN FP16 probability row then aliases the dead FP32 score storage.
template<int Threads,bool Async,bool Wait=true>__device__ __forceinline__ void load_half(const __half *x,__half *tile,int first,int head,int heads,int valid){
    constexpr int Vectors=Keys*Dim/8;
    #pragma unroll
    for(int i=threadIdx.x;i<Vectors;i+=Threads){
        const int token=first+i/32,d=(i%32)*8;const bool active=token<valid;
        auto *destination=tile+(i/32)*Pitch+d;
        const auto *source=active?x+(size_t(token)*heads+head)*Dim+d:x;
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
        if constexpr(Async){
            const unsigned shared=unsigned(__cvta_generic_to_shared(destination));const int bytes=active?16:0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(shared),"l"(source),"r"(bytes):"memory");
        }else
        #endif
        {const uint4 value=active?*reinterpret_cast<const uint4 *>(source):uint4{};*reinterpret_cast<uint4 *>(destination)=value;}
    }
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
    if constexpr(Async){
        asm volatile("cp.async.commit_group;":::"memory");
        if constexpr(Wait)asm volatile("cp.async.wait_group 0;":::"memory");
    }
    #endif
}
template<int Groups>__device__ void score_tile(CompactQueryTile<Groups> &t,int warp){
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
    wmma::store_matrix_sync(t.row[query*16].score+key*16,dots,ScorePitch,wmma::mem_row_major);
}
template<int Groups,bool Async,int Pipeline=0>__global__ __launch_bounds__(128*Groups,1) void rescale_skip_kernel(const float *query,const __half *key,const __half *value,
                                         float *out,PrefillAttentionShape s){
    constexpr int Rows=16*Groups,Threads=128*Groups,Warps=4*Groups;
    extern __shared__ __align__(32) unsigned char arena[];
    auto &t=*reinterpret_cast<CompactQueryTile<Groups> *>(arena);
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
    if constexpr(Pipeline==2)load_half<Threads,true,false>(key,t.kv,0,kh,s.kv_heads,s.valid);
    for(int first=0;first<end;first+=Keys){
        if constexpr(Pipeline==2)asm volatile("cp.async.wait_group 0;":::"memory");
        else load_half<Threads,Async>(key,t.kv,first,kh,s.kv_heads,s.valid);
        __syncthreads();
        score_tile(t,warp);__syncthreads();
        // All QK readers have retired. V can reuse K storage while softmax
        // and the register-only result rescaling proceed.
        if constexpr(Pipeline>0)load_half<Threads,true,false>(value,t.kv,first,kh,s.kv_heads,s.valid);
        for(int row=warp;row<Rows;row+=Warps){
            const int token0=first+lane,token1=token0+32,boundary=s.query_start+first_query+row;
            const bool query_active=first_query+row<s.tokens;
            const bool active0=query_active&&token0<end&&token0<=boundary;
            const bool active1=query_active&&token1<end&&token1<=boundary;
            const float score0=active0?t.row[row].score[lane]:-CUDART_INF_F;
            const float score1=active1?t.row[row].score[lane+32]:-CUDART_INF_F;
            __syncwarp(); // Every lane has read both score halves before this row changes lifetime.
            const float next=fmaxf(t.maximum[row],max_warp(fmaxf(score0,score1)));
            const float adjust=isfinite(t.maximum[row])?__expf(t.maximum[row]-next):0.f;
            const float p0=active0?__expf(score0-next):0.f,p1=active1?__expf(score1-next):0.f;
            const __half high0=__float2half_rn(p0),high1=__float2half_rn(p1);
            t.row[row].probability[lane]=high0;t.row[row].probability[lane+32]=high1;
            const float total=sum_warp(p0+p1);
            if(!lane){t.denominator[row]=fmaf(t.denominator[row],adjust,total);t.maximum[row]=next;t.adjust[row]=adjust;}
        }
        __syncthreads();
        const bool needs_rescale=__any_sync(0xffffffff,
            lane<16&&t.adjust[group*16+lane]!=1.f);
        if(needs_rescale){
            #pragma unroll
            for(int i=0;i<Fragment::num_elements;++i){
                const float adjust=t.adjust[group*16+((columns>>(4*i))&15)];
                #pragma unroll
                for(int f=0;f<4;++f)result[f].x[i]*=adjust;
            }
        }
        if constexpr(Pipeline>0)asm volatile("cp.async.wait_group 0;":::"memory");
        else load_half<Threads,Async>(value,t.kv,first,kh,s.kv_heads,s.valid);
        __syncthreads();
        #pragma unroll
        for(int offset=0;offset<Keys;offset+=16){
            // FP32 softmax statistics, single RN FP16 probability operand.
            // FP32 accumulation and the original256KV history seam remain.
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> high;
            wmma::load_matrix_sync(high,t.row[group*16].probability+offset,ProbPitch);
            #pragma unroll
            for(int f=0;f<4;++f){
                wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> a;
                wmma::load_matrix_sync(a,t.kv+offset*Pitch+((warp%4)*4+f)*16,Pitch);
                wmma::mma_sync(result[f],a,high,result[f]);
            }
        }
        __syncthreads();
        // All PV readers have retired. The FP32 history seam below touches
        // only row metadata/registers, so the next K tile can arrive here.
        if constexpr(Pipeline==2)if(first+Keys<end)
            load_half<Threads,true,false>(key,t.kv,first+Keys,kh,s.kv_heads,s.valid);
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
    // The final seam barrier has retired every denominator reader/writer.
    // Its storage now holds one RN reciprocal per row; the first store barrier
    // below publishes it before any output lane consumes the new lifetime.
    if(threadIdx.x<Rows)t.denominator[threadIdx.x]=__frcp_rn(t.denominator[threadIdx.x]);
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
                    out[index]=temporary[row*Dim+local]*t.denominator[qrow];
                    out[index+128]=temporary[row*Dim+local+128]*t.denominator[qrow];
                }
            }
        }
        __syncthreads();
    }
}
template<int Groups,bool Async,int Pipeline=0>void configure(){
    const int bytes=int(sizeof(CompactQueryTile<Groups>));int device=0;check_cuda(cudaGetDevice(&device),"compact attention device");
    const auto caps=capabilities(device);
    if(caps.major!=12||caps.minor!=0||caps.shared_per_block<size_t(bytes))throw std::runtime_error("compact attention SM120 shared admission");
    check_cuda(cudaFuncSetAttribute(rescale_skip_kernel<Groups,Async,Pipeline>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes),"compact attention shared");
    check_cuda(cudaFuncSetAttribute(rescale_skip_kernel<Groups,Async,Pipeline>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"compact attention carveout");
}
template<int Groups,bool Async,int Pipeline=0>void launch(const float *q,const __half *k,const __half *v,float *out,PrefillAttentionShape s,cudaStream_t stream){
    rescale_skip_kernel<Groups,Async,Pipeline><<<dim3(s.query_heads,(s.tokens+16*Groups-1)/(16*Groups)),128*Groups,sizeof(CompactQueryTile<Groups>),stream>>>(q,k,v,out,s);
}
template<int Groups,bool Async,int Pipeline=0>PrefillOnlineResources resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,rescale_skip_kernel<Groups,Async,Pipeline>),"compact attention resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,rescale_skip_kernel<Groups,Async,Pipeline>,128*Groups,sizeof(CompactQueryTile<Groups>)),"compact attention occupancy");
    return {a.numRegs,blocks,a.maxThreadsPerBlock,a.localSizeBytes,sizeof(CompactQueryTile<Groups>)};
}
}
void configure_rescale_skip_attention(){configure<4,true,2>();}
PrefillOnlineResources rescale_skip_attention_resources(){return resources<4,true,2>();}
void rescale_skip_attention(const float *q,const void *k,const void *v,float *out,void *workspace,size_t bytes,
                        PrefillAttentionShape s,cudaStream_t stream){
    if(!q||!out||!s.q8||s.dim!=Dim||s.query_heads<1||s.query_heads>256||s.kv_heads<1||s.query_heads%s.kv_heads||
       s.tokens<1||s.tokens>8192||s.query_start<0||s.valid!=s.query_start+s.tokens||s.valid>s.capacity||s.capacity>262144||
       !(s.scale>0)||!std::isfinite(s.scale))throw std::invalid_argument("PV reuse attention causal geometry");
    prefill_decode_kv(k,v,workspace,bytes,s,stream);
    auto *key=static_cast<const __half *>(workspace),*value=key+prefill_attention_workspace(s)/(2*sizeof(__half));
    launch<4,true,2>(q,key,value,out,s,stream);
    check_cuda(cudaGetLastError(),"PV reuse attention result");
}
}
