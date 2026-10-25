#include <strata/qwen35/native/attention_ram.hpp>
#include <strata/qwen35/native/attention_prefill.hpp>
#include <strata/qwen35/native/attention_verify.hpp>
#include <strata/qwen35/native/ffn.hpp>
#include <mma.h>
#include <cmath>
#include <math_constants.h>
#include "ram_score.hpp"
#include "ram_six.hpp"
#include "ram_score_policy.hpp"
#include <cstdint>
#include <limits>

namespace strata::qwen35::native {
namespace {
namespace wmma=nvcuda::wmma;
constexpr int Dim=256,Rows=16,Keys=32,Pitch=272,ProbPitch=48,ScorePitch=40,Partial=258;
constexpr int Group=6,LiveRows=Group;
__device__ float sum_warp(float x){
    for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;
}
__device__ float max_warp(float x){
    for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;
}
__device__ void load_q8(const unsigned char *source,__half *tile,int first,int head,int heads,int valid,int capacity,RamKvView ram,int logical_valid){
    // Four lanes load the same 34-byte Q8 block; each decodes eight codes.
    // Nothing is expanded to a global FP16 cache.
    for(int i=threadIdx.x;i<Keys*Dim/8;i+=128){
        const int token=first+i/32,block=(i%32)/4,offset=(i%4)*8;
        __align__(16) __half values[8];
        const int logical=token<valid?ram.sorted_ids[size_t(head)*ram.pages+token/RamPage]*RamPage+token%RamPage:logical_valid;
        if(token<valid&&logical<logical_valid){
            const int rank=token/RamPage,inpage=token%RamPage;
            const int physical=ram.selected_slot[size_t(head)*ram.slots+rank]*RamPage+inpage;
            const auto *p=source+(size_t(physical)*heads+head)*272+block*34;
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
template<bool CacheScores> __global__ void grouped_partial_attention(const float *query,const unsigned char *key,const unsigned char *value,
                                  float *partial,PrefillAttentionShape s,int partitions,const int *positions,int shift,int window,RamKvView ram,float *saved,int cache_keys){
    __shared__ __align__(32) __half q[Rows*Pitch],kv[Keys*Pitch],prob[2*Rows*ProbPitch];
    __shared__ __align__(32) float score[Rows*ScorePitch];
    __shared__ float maximum[Rows],denominator[Rows];
    s.query_start=positions[0]-shift;s.valid=s.query_start+s.tokens;
    const int head=blockIdx.x*Group,part=blockIdx.y,kh=head/(s.query_heads/s.kv_heads);
    const int live=Group*s.tokens;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if constexpr(CacheScores)saved+=(size_t(blockIdx.x)*partitions+part)*LiveRows*cache_keys;
    const int selected=min((s.valid+RamPage-1)/RamPage,ram.selected_slots)*RamPage;
    const int tiles=(selected+Keys-1)/Keys,span=(tiles+partitions-1)/partitions;
    const int begin=part*span*Keys,end=min(selected,begin+span*Keys);
    for(int i=threadIdx.x;i<Rows*Dim;i+=128){
        const int row=i/Dim;
        q[(i/Dim)*Pitch+i%Dim]=row<live?__hmul(__float2half_rn(query[(size_t(row%s.tokens)*s.query_heads+head+row/s.tokens)*Dim+i%Dim]),
                                __float2half_rn(s.scale)):__float2half(0);
    }
    if(threadIdx.x<Rows){maximum[threadIdx.x]=-CUDART_INF_F;denominator[threadIdx.x]=0;}
    __syncthreads();
    // Six query heads sharing one KV head fill six rows of the same WMMA tile.
    // The causal token position remains row % s.tokens.
    // Two passes avoid rescaling opaque WMMA accumulator layouts. FP32 tensor
    // partials are periodically merged with ordinary FP32 additions below.
    for(int first=begin;first<end;first+=Keys){
        load_q8(key,kv,first,kh,s.kv_heads,selected,s.capacity,ram,s.valid);__syncthreads();
        qk(q,kv,score,warp);__syncthreads();
        for(int row=warp;row<live;row+=4){
            const int selected_row=first+lane;
            const int token=selected_row<selected?ram.sorted_ids[size_t(kh)*ram.pages+selected_row/RamPage]*RamPage+selected_row%RamPage:s.valid;
            const float x=selected_row<end&&token<=s.query_start+row%s.tokens?score[row*ScorePitch+lane]:-CUDART_INF_F;
            if constexpr(CacheScores)if(selected_row<end)saved[size_t(row)*cache_keys+first-begin+lane]=score[row*ScorePitch+lane];
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
            load_q8(key,kv,first,kh,s.kv_heads,selected,s.capacity,ram,s.valid);__syncthreads();
            qk(q,kv,score,warp);__syncthreads();
        }
        for(int row=warp;row<Rows;row+=4){
            const int selected_row=first+lane;
            const int token=selected_row<selected?ram.sorted_ids[size_t(kh)*ram.pages+selected_row/RamPage]*RamPage+selected_row%RamPage:s.valid;
            const bool active=row<live&&selected_row<end&&token<=s.query_start+row%s.tokens;
            float p;
            if constexpr(CacheScores)p=active?__expf(saved[size_t(row)*cache_keys+first-begin+lane]-maximum[row]):0.0f;
            else p=active?__expf(score[row*ScorePitch+lane]-maximum[row]):0.0f;
            const __half high=__float2half_rn(p);
            prob[row*ProbPitch+lane]=high;
            prob[Rows*ProbPitch+row*ProbPitch+lane]=__float2half_rn(p-__half2float(high));
            const float total=sum_warp(p);if(!lane)denominator[row]+=total;
        }
        __syncthreads();load_q8(value,kv,first,kh,s.kv_heads,selected,s.capacity,ram,s.valid);__syncthreads();
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

RamScoreLayout ram_score_layout(RamKvView r,int heads,int parts){
    if(heads!=24||r.heads!=4||parts!=64||r.selected_tokens!=8192||r.selected_slots!=512||
       r.resident!=32768||r.slots!=2048||r.pages<2049||r.pages>4096||r.host_capacity!=r.pages*RamPage)
        throw std::invalid_argument("RAM score fixed current geometry/capacity");
    const size_t partial=size_t(heads)*parts*Partial*4,offset=(partial+127)&~size_t(127);
    const int stride=Keys*((((r.selected_tokens+Keys-1)/Keys)+parts-1)/parts);
    const size_t scores=size_t(heads/Group)*parts*LiveRows*stride*4;
    return {partial,offset,scores,offset+scores,stride};
}
namespace {
struct Range {uintptr_t first,last;};
Range range(const void *p,size_t n,size_t alignment){
    const auto a=reinterpret_cast<uintptr_t>(p);
    if(!p||a%alignment||n>std::numeric_limits<uintptr_t>::max()-a)throw std::invalid_argument("RAM score pointer/alignment/overflow");
    return {a,a+n};
}
bool overlap(Range a,Range b){return a.first<b.last&&b.first<a.last;}
}
void ram_score_validate(const float *q,RamKvView r,float *out,void *work,size_t bytes,const int *positions,int tokens,int heads,int parts){
    const auto l=ram_score_layout(r,heads,parts);
    if(tokens!=3||bytes<l.total_bytes)throw std::invalid_argument("RAM score span/arena");
    const size_t vectors=size_t(tokens)*heads*Dim*4,pool=size_t(r.resident)*r.heads*272,host=size_t(r.host_capacity)*r.heads*272;
    const Range live[]={range(q,vectors,4),range(out,vectors,4),range(positions,tokens*4,4),range(work,bytes,16),
        range(r.key,pool,2),range(r.value,pool,2),range(r.host_key,host,2),range(r.host_value,host,2)};
    for(int i=0;i<8;++i)for(int j=i+1;j<8;++j)if(overlap(live[i],live[j]))throw std::invalid_argument("RAM score live operand alias");
    const Range arena=live[3];
    const auto disjoint=[&](const void *p,size_t n,size_t a){if(overlap(arena,range(p,n,a))||overlap(live[1],range(p,n,a)))throw std::invalid_argument("RAM score directory/arena alias");};
    for(const auto p:{r.page_ids,r.sorted_ids,r.page_slot})disjoint(p,size_t(r.heads)*r.pages*4,4);
    for(const auto p:{r.slot_page,r.selected_slot,r.miss,r.marked,r.miss_rank,r.free_slot,r.last_use})disjoint(p,size_t(r.heads)*r.slots*4,4);
    for(const auto p:{r.scores,r.sorted_scores})disjoint(p,size_t(r.heads)*r.pages*4,4);
    disjoint(r.offsets,size_t(r.heads+1)*4,4);disjoint(r.sort_work,r.sort_bytes,16);
    disjoint(r.bounds,size_t(r.heads)*r.pages*512*2,2);disjoint(r.epoch,r.heads*4,4);disjoint(r.counters,sizeof(RamKvCounters),8);
}
RamScoreResources ram_score_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,grouped_partial_attention<true>),"RAM score resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,grouped_partial_attention<true>,128,0),"RAM score occupancy");
    return {a.numRegs,blocks,int(a.sharedSizeBytes),int(a.localSizeBytes)};
}
void ram_score_attention(const float *q,RamKvView ram,float *out,void *work,size_t bytes,const int *positions,int shift,int tokens,int heads,int parts,cudaStream_t stream){
    ram_score_validate(q,ram,out,work,bytes,positions,tokens,heads,parts);const auto l=ram_score_layout(ram,heads,parts);
    auto *scores=reinterpret_cast<float *>(static_cast<unsigned char *>(work)+l.score_offset);
    for(int token=0;token<tokens;++token){
        const float *query=q+size_t(token)*heads*Dim;float *output=out+size_t(token)*heads*Dim;
        ++ram_score_request().submissions;
        // Exact baseline selector and mutable residency; no selected-ID remapping or reuse across queries.
        ram_select(query,ram,positions+token,shift,heads,stream);
        PrefillAttentionShape s{Dim,heads,ram.heads,1,0,1,ram.resident,1.f/16.f,{},true};
        grouped_partial_attention<true><<<dim3(heads/Group,parts),128,0,stream>>>(query,ram.key,ram.value,static_cast<float *>(work),s,parts,
            positions+token,shift,ram.resident,ram,scores,l.cache_keys);
        check_cuda(cudaGetLastError(),"RAM score cached verify");
        verify_attention_merge(static_cast<const float *>(work),output,heads,1,parts,stream);
    }
}
void ram_verify_six(const float *q,RamKvView ram,float *out,void *scratch,size_t bytes,
                    const int *positions,int shift,int heads,int partitions,cudaStream_t stream){
    if(!q||!out||!scratch||!positions||bytes<size_t(heads)*partitions*258*4||(heads/ram.heads)%6||heads%ram.heads||partitions<1||partitions>128)
        throw std::invalid_argument("invalid RAM selected verify geometry");
    PrefillAttentionShape s{256,heads,ram.heads,1,0,1,ram.resident,1.f/16.f,{},true};
    grouped_partial_attention<false><<<dim3(heads/6,partitions),128,0,stream>>>(q,ram.key,ram.value,static_cast<float *>(scratch),
        s,partitions,positions,shift,ram.resident,ram,nullptr,0);
    check_cuda(cudaGetLastError(),"RAM selected per-query causal verify");
    verify_attention_merge(static_cast<const float *>(scratch),out,heads,1,partitions,stream);
}
RamSixResources ram_six_resources(){
    cudaFuncAttributes a{};check_cuda(cudaFuncGetAttributes(&a,grouped_partial_attention<false>),"RAM group6 resources");
    return {a.numRegs,int(a.localSizeBytes),int(a.sharedSizeBytes)};
}
void ram_attention_six(const float *q,RamKvView ram,float *out,void *work,size_t bytes,const int *positions,
                   int shift,int tokens,int heads,DecodeRounding rounding,int partitions,cudaStream_t stream){
    if(tokens==1||(heads/ram.heads)%6){ram_attention(q,ram,out,work,bytes,positions,shift,tokens,heads,rounding,stream);return;}
    if(!q||!out||!work||!positions||tokens<2||tokens>3||bytes<size_t(heads)*partitions*258*4)
        throw std::invalid_argument("invalid group6 RAM verify workspace");
    for(int token=0;token<tokens;++token){
        const float *query=q+size_t(token)*heads*256;float *output=out+size_t(token)*heads*256;
        ram_select(query,ram,positions+token,shift,heads,stream);
        ram_verify_six(query,ram,output,work,bytes,positions+token,shift,heads,partitions,stream);
    }
}

}
