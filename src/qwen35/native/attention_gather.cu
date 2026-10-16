#include "strata/qwen35/native/attention_gather.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <mma.h>
#include <cmath>
#include <math_constants.h>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Dim=256,Keys=32,Pitch=Dim+16,ProbPitch=Keys+16,ScorePitch=Keys+8;
template<int Groups>struct GatherQueryTile {
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
    uint4 pending[Chunks];
    #pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk){
        const int i=threadIdx.x+chunk*Threads,token=first+i/32,d=(i%32)*8;
        pending[chunk]=uint4{};
        if(i<Vectors&&token<valid)pending[chunk]=*reinterpret_cast<const uint4 *>(x+(size_t(token)*heads+head)*Dim+d);
    }
    asm volatile("" ::: "memory");
    #pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk){
        const int i=threadIdx.x+chunk*Threads;
        if(i<Vectors)*reinterpret_cast<uint4 *>(tile+(i/32)*Pitch+(i%32)*8)=pending[chunk];
    }
}
template<int Groups>__device__ void score_tile(GatherQueryTile<Groups> &t,int warp){
    if(warp>=2*Groups)return;
    const int query=warp/2,key=warp%2;
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
template<int Groups>__global__ void gather_kernel(const float *query,const __half *key,const __half *value,
                                         float *out,PrefillAttentionShape s){
    constexpr int Rows=16*Groups,Threads=128*Groups,Warps=4*Groups;
    extern __shared__ __align__(32) unsigned char arena[];
    auto &t=*reinterpret_cast<GatherQueryTile<Groups> *>(arena);
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
    for(int first=0;first<end;first+=Keys){
        load_half<Threads>(key,t.kv,first,kh,s.kv_heads,s.valid);__syncthreads();
        score_tile(t,warp);__syncthreads();
        for(int row=warp;row<Rows;row+=Warps){
            const int token=first+lane;
            const bool active=first_query+row<s.tokens&&token<end&&token<=s.query_start+first_query+row;
            const float score=active?t.score[row*ScorePitch+lane]:-CUDART_INF_F;
            const float next=fmaxf(t.maximum[row],max_warp(score));
            const float adjust=isfinite(t.maximum[row])?__expf(t.maximum[row]-next):0.f;
            const float p=active?__expf(score-next):0.f;
            const __half high=__float2half_rn(p);
            t.prob[row*ProbPitch+lane]=high;t.prob[Rows*ProbPitch+row*ProbPitch+lane]=__float2half_rn(p-__half2float(high));
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
        if(first/Keys%8==7||first+Keys>=end){
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
    const int bytes=int(sizeof(GatherQueryTile<Groups>));int device=0;check_cuda(cudaGetDevice(&device),"gather-load online PP setup device");
    if(capabilities(device).shared_per_block<size_t(bytes))throw std::runtime_error("gather-load online PP exceeds shared capacity");
    check_cuda(cudaFuncSetAttribute(gather_kernel<Groups>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes),"gather-load online PP shared reservation");
    check_cuda(cudaFuncSetAttribute(gather_kernel<Groups>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"gather-load online PP shared partition");
}
template<int Groups>void launch(const float *q,const __half *k,const __half *v,float *out,PrefillAttentionShape s,cudaStream_t stream){
    gather_kernel<Groups><<<dim3(s.query_heads,(s.tokens+16*Groups-1)/(16*Groups)),128*Groups,sizeof(GatherQueryTile<Groups>),stream>>>(q,k,v,out,s);
}
template<int Groups>PrefillOnlineResources resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,gather_kernel<Groups>),"gather-load online PP resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,gather_kernel<Groups>,128*Groups,sizeof(GatherQueryTile<Groups>)),"gather-load online PP occupancy bound");
    return {a.numRegs,blocks,a.maxThreadsPerBlock,a.localSizeBytes,sizeof(GatherQueryTile<Groups>)};
}
}
PrefillOnlineResources prefill_gather_resources(int rows){
    if(rows==16)return resources<1>();if(rows==32)return resources<2>();if(rows==48)return resources<3>();
    throw std::invalid_argument("unsupported PP query tile");
}
size_t prefill_gather_shared_bytes(int rows){
    if(rows==16)return sizeof(GatherQueryTile<1>);if(rows==32)return sizeof(GatherQueryTile<2>);if(rows==48)return sizeof(GatherQueryTile<3>);
    throw std::invalid_argument("unsupported PP query tile");
}
void configure_prefill_gather(int rows){
    if(rows==16)configure<1>();else if(rows==32)configure<2>();else if(rows==48)configure<3>();
    else throw std::invalid_argument("unsupported PP query tile");
}
void prefill_attention_gather(const float *q,const void *k,const void *v,float *out,void *workspace,size_t bytes,
                            PrefillAttentionShape s,int rows,cudaStream_t stream){
    prefill_gather_shared_bytes(rows);
    if(!q||!out||!s.q8||s.dim!=Dim||s.query_heads<1||s.query_heads>256||s.kv_heads<1||
       s.query_heads%s.kv_heads||s.tokens<1||s.tokens>8192||s.query_start<0||s.valid!=s.query_start+s.tokens||
       s.valid>s.capacity||s.capacity>262144||!(s.scale>0)||!std::isfinite(s.scale))
        throw std::invalid_argument("invalid gather-load online PP causal geometry");
    prefill_decode_kv(k,v,workspace,bytes,s,stream);
    auto *key=static_cast<const __half *>(workspace),*value=key+prefill_attention_workspace(s)/(2*sizeof(__half));
    if(rows==16)launch<1>(q,key,value,out,s,stream);else if(rows==32)launch<2>(q,key,value,out,s,stream);else if(rows==48)launch<3>(q,key,value,out,s,stream);

    check_cuda(cudaGetLastError(),"owned gather-load online-softmax FP32 PP attention");
}
}
