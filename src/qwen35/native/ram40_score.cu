#include <strata/qwen35/native/attention_ram.hpp>
#include <strata/qwen35/native/attention_prefill.hpp>
#include <strata/qwen35/native/attention_verify.hpp>
#include <strata/qwen35/native/ffn.hpp>
#include <mma.h>
#include <cmath>
#include <algorithm>
#include <math_constants.h>
#include "ram40_score.hpp"
#include "ram_batch_select_policy.hpp"
#include "ram40_triple_wave_policy.hpp"
#include "ram40_prefix512_policy.hpp"
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cstdint>
#include <limits>

namespace strata::qwen35::native {
namespace {
namespace prefix512_detail {
constexpr int Threads=256,Items=10,Pages=2560,Keep=512,Heads=4;
// CUB FP32 descending radix order: signed zeros compare equal; NaNs retain transformed bit order.
__device__ unsigned ordered(float x){unsigned u=__float_as_uint(x);if(!(u&0x7fffffffU))u=0;return u^((u&0x80000000U)?0xffffffffU:0x80000000U);}
__device__ unsigned block_prefix(unsigned n,unsigned*warp_sum){
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;unsigned scan=n;
    for(int shift=1;shift<32;shift*=2){const unsigned x=__shfl_up_sync(0xffffffffU,scan,shift);if(lane>=shift)scan+=x;}
    if(lane==31)warp_sum[warp]=scan;__syncthreads();unsigned before=scan-n;
    for(int w=0;w<warp;++w)before+=warp_sum[w];__syncthreads();return before;
}
__global__ void prefix_sort(RamKvView v,const int*positions,const int*base_epoch,const float*scores,float*sorted_scores,int*sorted_ids){
    using Sort=cub::BlockRadixSort<float,Threads,2,int>;
    __shared__ typename Sort::TempStorage sort_temp;__shared__ unsigned warp_sum[8];
    __shared__ unsigned histogram[8][16],prefix,mask,needed;
    __shared__ float packed_keys[Keep];__shared__ int packed_ids[Keep];
    const int h=blockIdx.x,t=blockIdx.y,tid=threadIdx.x;const size_t head=size_t(h)*Pages,base=size_t(t)*Heads*Pages+head;
    const int count=min((positions[t]+RamPage)/RamPage,v.selected_slots),epoch=base_epoch[h]+t+1;
    float keys[Items];unsigned bits[Items];int ids[Items];
    #pragma unroll
    for(int i=0;i<Items;++i){const int page=tid*Items+i;keys[i]=scores[base+page];bits[i]=ordered(keys[i]);ids[i]=v.page_ids[head+page];if(t==2)v.scores[head+page]=keys[i];}
    if(!tid){prefix=0;mask=0;needed=Keep;}__syncthreads();
    // One fixed MSB nibble threshold pass. Only the boundary bucket participates in the next digit.
    for(int shift=28;shift>=0;shift-=4){
        if(tid<128)histogram[tid/16][tid%16]=0;__syncthreads();
        #pragma unroll
        for(int i=0;i<Items;++i)if((bits[i]&mask)==prefix)atomicAdd(&histogram[tid/32][(bits[i]>>shift)&15],1U);
        __syncthreads();
        if(!tid){unsigned before=0;int chosen=0;
            for(int bin=15;bin>=0;--bin){unsigned n=0;for(int warp=0;warp<8;++warp)n+=histogram[warp][bin];
                if(before+n>=needed){chosen=bin;needed-=before;break;}before+=n;}
            prefix|=unsigned(chosen)<<shift;mask|=15U<<shift;
        }__syncthreads();
    }
    unsigned ties=0;
    #pragma unroll
    for(int i=0;i<Items;++i)ties+=bits[i]==prefix;
    const unsigned tie_before=block_prefix(ties,warp_sum);
    unsigned chosen=0,local_ties=0;
    #pragma unroll
    for(int i=0;i<Items;++i){const bool same=bits[i]==prefix;chosen+=bits[i]>prefix||(same&&tie_before+local_ties<needed);local_ties+=same;}
    const unsigned offset=block_prefix(chosen,warp_sum);local_ties=0;unsigned written=0;
    #pragma unroll
    for(int i=0;i<Items;++i){const bool same=bits[i]==prefix,keep=bits[i]>prefix||(same&&tie_before+local_ties<needed);local_ties+=same;
        if(keep){packed_keys[offset+written]=keys[i];packed_ids[offset+written]=ids[i];++written;}}
    __syncthreads();float top[2]={packed_keys[tid*2],packed_keys[tid*2+1]};int selected[2]={packed_ids[tid*2],packed_ids[tid*2+1]};
    Sort(sort_temp).SortDescending(top,selected);__syncthreads();
    if(t==2)for(int page=tid;page<Pages;page+=Threads)v.marked[head+page]=0;__syncthreads();
    #pragma unroll
    for(int i=0;i<2;++i){const int rank=tid*2+i,page=selected[i];sorted_scores[base+rank]=top[i];sorted_ids[base+rank]=page;
        if(t==2){v.sorted_scores[head+rank]=top[i];v.sorted_ids[head+rank]=page;}
        if(rank<count){atomicMax(v.last_use+head+page,epoch);if(t==2){v.selected_slot[head+rank]=page;v.miss[head+rank]=0;v.marked[head+page]=1;}}}
    if(!tid){atomicAdd(&v.counters->calls,1ULL);atomicAdd(&v.counters->selected_pages,(unsigned long long)count);atomicAdd(&v.counters->hits,(unsigned long long)count);if(t==2)v.epoch[h]=base_epoch[h]+3;}
}
} // namespace prefix512_detail

namespace batch_select_detail {
constexpr int Heads=24,KvHeads=4,Pages=2560,Slots=2560,Tokens=3,Dim=256;
__global__ void batch_scores(const float*query,RamKvView v,const int*positions,float*scores,int*base_epoch){
    const int page=blockIdx.x,h=blockIdx.y,t=blockIdx.z,d=threadIdx.x;
    if(!page&&!t&&!d)base_epoch[h]=v.epoch[h]; // Complete before the dependent sort kernel starts.
    const int valid=positions[t]+1,live=(valid+RamPage-1)/RamPage;const size_t i=size_t(h)*v.pages+page;
    float*dst=scores+size_t(t)*KvHeads*Pages+i;
    if(page>=live){if(!d)*dst=-CUDART_INF_F;return;}
    if(page<4||page>=live-64){if(!d)*dst=CUDART_INF_F;return;}
    const float lo=__half2float(v.bounds[i*512+d]),hi=__half2float(v.bounds[i*512+256+d]);
    __shared__ float partial[6][8];
    for(int q=0;q<6;++q){
        const float x=query[(size_t(t)*Heads+h*6+q)*Dim+d];float score=fmaxf(x*lo,x*hi);
        for(int step=16;step;step/=2)score+=__shfl_xor_sync(0xffffffff,score,step);
        if(!(d%32))partial[q][d/32]=score;
    }
    __syncthreads();
    if(d<32){float result=-CUDART_INF_F;
        for(int q=0;q<6;++q){float score=d<8?partial[q][d]:0.f;
            for(int step=16;step;step/=2)score+=__shfl_xor_sync(0xffffffff,score,step);
            if(!d)result=fmaxf(result,score);}
        if(!d)*dst=result;
    }
}
__global__ void batch_sort(RamKvView v,const int*positions,const int*base_epoch,const float*scores,float*sorted_scores,int*sorted_ids){
    constexpr int Threads=256,Items=16;
    using LoadKeys=cub::BlockLoad<float,Threads,Items,cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using LoadIds=cub::BlockLoad<int,Threads,Items,cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using Sort=cub::BlockRadixSort<float,Threads,Items,int>;
    __shared__ union Scratch {typename LoadKeys::TempStorage keys;typename LoadIds::TempStorage ids;typename Sort::TempStorage sort;} temp;
    const int h=blockIdx.x,t=blockIdx.y;const size_t head=size_t(h)*Pages,base=size_t(t)*KvHeads*Pages+head;
    const int count=min((positions[t]+RamPage)/RamPage,v.selected_slots),epoch=base_epoch[h]+t+1;
    float keys[Items];int ids[Items];
    LoadKeys(temp.keys).Load(scores+base,keys,Pages,__int_as_float(0xffffffff));__syncthreads();
    LoadIds(temp.ids).Load(v.page_ids+head,ids,Pages,-1);__syncthreads();Sort(temp.sort).SortDescending(keys,ids);
    if(t==2)for(int page=threadIdx.x;page<Pages;page+=Threads)v.marked[head+page]=0;
    __syncthreads();
    #pragma unroll
    for(int i=0;i<Items;++i){const int rank=threadIdx.x*Items+i;
        if(rank<Pages){sorted_scores[base+rank]=keys[i];sorted_ids[base+rank]=ids[i];
            if(t==2){v.scores[head+rank]=scores[base+rank];v.sorted_scores[head+rank]=keys[i];v.sorted_ids[head+rank]=ids[i];}}
        if(rank<count){const int page=ids[i];
            atomicMax(v.last_use+head+page,epoch);
            if(t==2){v.selected_slot[head+rank]=page;v.miss[head+rank]=0;v.marked[head+page]=1;}
        }
    }
    if(!threadIdx.x){atomicAdd(&v.counters->calls,1ULL);atomicAdd(&v.counters->selected_pages,(unsigned long long)count);
        atomicAdd(&v.counters->hits,(unsigned long long)count);if(t==2)v.epoch[h]=base_epoch[h]+Tokens;}
}
}
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
template<bool CacheScores,bool Batched=false> __global__ void ram40_grouped_partial_attention(const float *query,const unsigned char *key,const unsigned char *value,
                                  float *partial,PrefillAttentionShape s,int partitions,const int *positions,int shift,int window,RamKvView ram,float *saved,int cache_keys){
    int batch_token=0;
    if constexpr(Batched){batch_token=blockIdx.z;
        query+=size_t(batch_token)*s.query_heads*Dim;positions+=batch_token;
        ram.scores+=size_t(batch_token)*4*2560;ram.sorted_scores+=size_t(batch_token)*4*2560;
        ram.sorted_ids+=size_t(batch_token)*4*2560;ram.selected_slot=ram.sorted_ids;
        saved+=size_t(batch_token)*4*partitions*LiveRows*cache_keys;
    }
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
        float *dst;
        if constexpr(Batched)dst=partial+((size_t(h)*partitions+part)*3+batch_token)*Partial;
        else dst=partial+((size_t(h)*partitions+part)*s.tokens+t)*Partial;
        #pragma unroll
        for(int d=0;d<2;++d)dst[threadIdx.x+d*128]=carry[row*2+d];
        if(!threadIdx.x){dst[Dim]=maximum[row];dst[Dim+1]=denominator[row];}
    }
}

}

Ram40ScoreLayout ram40_score_layout(RamKvView r,int heads,int parts){
    if(heads!=24||r.heads!=4||parts!=64||r.selected_tokens!=8192||r.selected_slots!=512||r.keep_queries!=3||
       r.resident!=40960||r.slots!=2560||r.pages!=2560||r.host_capacity!=40960||
       r.host_key!=r.key||r.host_value!=r.value)
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
void ram40_score_validate(const float *q,RamKvView r,float *out,void *work,size_t bytes,const int *positions,int tokens,int heads,int parts){
    const auto l=ram40_score_layout(r,heads,parts);
    if(tokens!=3||bytes<l.total_bytes)throw std::invalid_argument("RAM score span/arena");
    const size_t vectors=size_t(tokens)*heads*Dim*4,pool=size_t(r.resident)*r.heads*272;
    const Range live[]={range(q,vectors,4),range(out,vectors,4),range(positions,tokens*4,4),range(work,bytes,16),
        range(r.key,pool,2),range(r.value,pool,2)};
    for(int i=0;i<6;++i)for(int j=i+1;j<6;++j)if(overlap(live[i],live[j]))throw std::invalid_argument("RAM score live operand alias");
    const Range arena=live[3];
    const auto disjoint=[&](const void *p,size_t n,size_t a){if(overlap(arena,range(p,n,a))||overlap(live[1],range(p,n,a)))throw std::invalid_argument("RAM score directory/arena alias");};
    for(const auto p:{r.page_ids,r.sorted_ids,r.page_slot})disjoint(p,size_t(r.heads)*r.pages*4,4);
    for(const auto p:{r.slot_page,r.selected_slot,r.miss,r.marked,r.miss_rank,r.free_slot,r.last_use})disjoint(p,size_t(r.heads)*r.slots*4,4);
    for(const auto p:{r.scores,r.sorted_scores})disjoint(p,size_t(r.heads)*r.pages*4,4);
    disjoint(r.offsets,size_t(r.heads+1)*4,4);disjoint(r.sort_work,r.sort_bytes,16);
    disjoint(r.bounds,size_t(r.heads)*r.pages*512*2,2);disjoint(r.epoch,r.heads*4,4);disjoint(r.counters,sizeof(RamKvCounters),8);
}
Ram40ScoreResources ram40_score_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,ram40_grouped_partial_attention<true>),"RAM score resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,ram40_grouped_partial_attention<true>,128,0),"RAM score occupancy");
    return {a.numRegs,blocks,int(a.sharedSizeBytes),int(a.localSizeBytes),a.binaryVersion};
}
Ram40TripleWaveResources ram40_triple_wave_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,ram40_grouped_partial_attention<true,true>),"triple-wave resources");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,ram40_grouped_partial_attention<true,true>,128,0),"triple-wave occupancy");
    return {a.numRegs,blocks,int(a.sharedSizeBytes),int(a.localSizeBytes),a.binaryVersion};
}
Ram40Prefix512Resources ram40_prefix512_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,prefix512_detail::prefix_sort),"prefix512 attrs");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,prefix512_detail::prefix_sort,256,0),"prefix512 occupancy");
    return {a.numRegs,blocks,int(a.sharedSizeBytes),int(a.localSizeBytes),a.binaryVersion};
}
RamBatchSelectResources ram_batch_select_resources(){
    cudaFuncAttributes a{},b{};int x=0,y=0;
    check_cuda(cudaFuncGetAttributes(&a,batch_select_detail::batch_scores),"batch-select score attrs");
    check_cuda(cudaFuncGetAttributes(&b,batch_select_detail::batch_sort),"batch-select sort attrs");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&x,batch_select_detail::batch_scores,256,0),"batch-select score occupancy");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&y,batch_select_detail::batch_sort,256,0),"batch-select sort occupancy");
    return {a.numRegs,b.numRegs,x,y,int(a.localSizeBytes+b.localSizeBytes),std::min(a.binaryVersion,b.binaryVersion)};
}
void ram40_score_attention(const float *q,RamKvView ram,float *out,void *work,size_t bytes,const int *positions,int shift,int tokens,int heads,int parts,cudaStream_t stream){
    ram40_score_validate(q,ram,out,work,bytes,positions,tokens,heads,parts);const auto l=ram40_score_layout(ram,heads,parts);
    auto *scores=reinterpret_cast<float *>(static_cast<unsigned char *>(work)+l.score_offset);
    const bool batch=reinterpret_cast<uintptr_t>(work)%128==0&&use_ram_batch_select(ram,tokens,heads,parts,shift,bytes);
    const bool triple=use_ram40_triple_wave(batch,ram,tokens,heads,parts,shift,bytes);
    const bool prefix512=use_ram40_prefix512(triple,ram,tokens,heads,parts,shift,bytes);
    int*base=nullptr;float*query_scores=nullptr,*sorted_scores=nullptr;int*sorted_ids=nullptr;
    if(batch){
        if(l.total_bytes!=2371584)throw std::logic_error("batch-select retained layout changed");
        base=reinterpret_cast<int*>(static_cast<unsigned char*>(work)+l.total_bytes);constexpr size_t plane=3*4*2560;
        query_scores=reinterpret_cast<float*>(base+32);sorted_scores=query_scores+plane;sorted_ids=reinterpret_cast<int*>(sorted_scores+plane);
        batch_select_detail::batch_scores<<<dim3(2560,4,3),256,0,stream>>>(q,ram,positions,query_scores,base);
        check_cuda(cudaGetLastError(),"batch-select score launch");
        if(prefix512)
            prefix512_detail::prefix_sort<<<dim3(4,3),256,0,stream>>>(ram,positions,base,query_scores,sorted_scores,sorted_ids);
        else
            batch_select_detail::batch_sort<<<dim3(4,3),256,0,stream>>>(ram,positions,base,query_scores,sorted_scores,sorted_ids);
        check_cuda(cudaGetLastError(),"batch-select sort launch");
    }
    if(triple){
        auto selected=ram;selected.scores=query_scores;selected.sorted_scores=sorted_scores;
        selected.sorted_ids=sorted_ids;selected.selected_slot=sorted_ids;
        auto*partial=reinterpret_cast<float*>(static_cast<unsigned char*>(work)+TriplePartialOffset);
        auto*saved=reinterpret_cast<float*>(static_cast<unsigned char*>(work)+TripleScoreOffset);
        PrefillAttentionShape s{Dim,heads,ram.heads,1,0,1,ram.resident,1.f/16.f,{},true};
        ram40_grouped_partial_attention<true,true><<<dim3(heads/Group,parts,3),128,0,stream>>>(q,ram.key,ram.value,partial,s,parts,
            positions,shift,ram.resident,selected,saved,l.cache_keys);
        check_cuda(cudaGetLastError(),"RAM40 triple-wave cached verify");
        verify_attention_merge(partial,out,heads,3,parts,stream);
        return;
    }
    for(int token=0;token<tokens;++token){
        const float *query=q+size_t(token)*heads*Dim;float *output=out+size_t(token)*heads*Dim;
        // Exact baseline selector and mutable residency; no selected-ID remapping or reuse across queries.
        auto selected=ram;
        if(batch){selected.scores=query_scores+size_t(token)*4*2560;selected.sorted_scores=sorted_scores+size_t(token)*4*2560;
            selected.sorted_ids=sorted_ids+size_t(token)*4*2560;selected.selected_slot=selected.sorted_ids;}
        else ram_select(query,ram,positions+token,shift,heads,stream);
        PrefillAttentionShape s{Dim,heads,ram.heads,1,0,1,ram.resident,1.f/16.f,{},true};
        ram40_grouped_partial_attention<true><<<dim3(heads/Group,parts),128,0,stream>>>(query,ram.key,ram.value,static_cast<float *>(work),s,parts,
            positions+token,shift,ram.resident,selected,scores,l.cache_keys);
        check_cuda(cudaGetLastError(),"RAM score cached verify");
        verify_attention_merge(static_cast<const float *>(work),output,heads,1,parts,stream);
    }
}
}
