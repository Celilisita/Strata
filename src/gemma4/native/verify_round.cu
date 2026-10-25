#include "strata/gemma4/native/verify_round.hpp"
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace strata::gemma4::native {
namespace {
void checked(cudaError_t error,const char* where){
    if(error!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(error));
}
struct Candidate {float score;int id;};
__device__ Candidate better(Candidate a,Candidate b){
    return b.score>a.score||(b.score==a.score&&b.id<a.id)?b:a;
}
__device__ Candidate reduce(Candidate value){
    for(int distance=16;distance;distance/=2){
        Candidate other{__shfl_down_sync(0xffffffff,value.score,distance),
            __shfl_down_sync(0xffffffff,value.id,distance)};
        if(threadIdx.x%32+distance<32)value=better(value,other);
    }
    return value;
}
__global__ void form(const int* pending,const int* first,const int* second,
    int* inputs,int rows,int vocabulary,unsigned* invalid){
    if(threadIdx.x>=rows)return;
    const int index=threadIdx.x,id=index==0?*pending:(index==1?*first:*second);
    const bool valid=id>=0&&id<vocabulary;
    if(!valid)atomicOr(invalid,16u);
    inputs[index]=valid?id:0;
}
__global__ void partial_pick(float* logits,Candidate* partial,int rows,int vocabulary,
    float cap,unsigned* invalid){
    __shared__ Candidate warps[8];
    const int row=blockIdx.y,index=blockIdx.x*256+threadIdx.x;
    Candidate value{-INFINITY,INT_MAX};
    if(index<vocabulary){
        const size_t at=size_t(row)*vocabulary+index;const float raw=logits[at];
        const float capped=cap*tanhf(raw/cap);logits[at]=capped;
        if(!isfinite(raw)||!isfinite(capped))atomicOr(invalid,16u);
        else value={capped,index};
    }
    value=reduce(value);if(!(threadIdx.x%32))warps[threadIdx.x/32]=value;
    __syncthreads();
    if(threadIdx.x<32){value=threadIdx.x<8?warps[threadIdx.x]:Candidate{-INFINITY,INT_MAX};
        value=reduce(value);if(!threadIdx.x)partial[size_t(row)*gridDim.x+blockIdx.x]=value;}
}
__global__ void final_pick(const Candidate* partial,int* picks,int blocks,unsigned* invalid){
    __shared__ Candidate warps[8];const int row=blockIdx.x;
    Candidate value{-INFINITY,INT_MAX};
    for(int i=threadIdx.x;i<blocks;i+=256)value=better(value,partial[size_t(row)*blocks+i]);
    value=reduce(value);if(!(threadIdx.x%32))warps[threadIdx.x/32]=value;
    __syncthreads();
    if(threadIdx.x<32){value=threadIdx.x<8?warps[threadIdx.x]:Candidate{-INFINITY,INT_MAX};
        value=reduce(value);if(!threadIdx.x){
            const bool valid=value.id!=INT_MAX&&isfinite(value.score);
            if(!valid)atomicOr(invalid,16u);picks[row]=valid?value.id:0;
        }}
}
__global__ void commit(const int* inputs,const int* picks,const float* features,
    float* seed,int* pending,int* position,VerifyResponse* response,int rows,
    int hidden,int vocabulary,int capacity,VerifyPolicy policy,unsigned* invalid){
    __shared__ int accepted,matched,count,valid,start,reason;
    if(!threadIdx.x){
        accepted=matched=count=0;start=*position;reason=int(VerifyStop::Continue);
        valid=!*invalid&&start>=0&&start<=capacity&&*policy.remaining>=0&&
            *pending>=0&&*pending<vocabulary;
        bool pending_eog=false;
        for(int i=0;i<policy.eog_count;++i){const int id=policy.eog_ids[i];
            valid=valid&&id>=0&&id<vocabulary;pending_eog=pending_eog||id==*pending;}
        if(valid&&pending_eog)reason=int(VerifyStop::Eog);
        else if(valid&&!*policy.remaining)reason=int(VerifyStop::Budget);
        else if(valid){
            valid=start<=capacity-rows&&inputs[0]==*pending;
            for(int i=0;i<rows;++i)
                valid=valid&&inputs[i]>=0&&inputs[i]<vocabulary&&picks[i]>=0&&picks[i]<vocabulary;
            if(valid){
                while(matched<rows-1&&inputs[matched+1]==picks[matched])++matched;
                count=matched+1;if(count>*policy.remaining)count=*policy.remaining;
                for(int i=0;i<count&&reason==int(VerifyStop::Continue);++i)
                    for(int j=0;j<policy.eog_count;++j)if(picks[i]==policy.eog_ids[j]){
                        count=i+1;reason=int(VerifyStop::Eog);break;}
                if(reason==int(VerifyStop::Continue)&&count==*policy.remaining)
                    reason=int(VerifyStop::Budget);
                accepted=count-1;
            }
        }
        if(!valid)atomicOr(invalid,16u);
    }
    __syncthreads();
    if(valid&&count){
        for(int i=threadIdx.x;i<rows*hidden;i+=256)
            if(!isfinite(features[i]))atomicOr(invalid,16u);
    }
    __syncthreads();
    if(!*invalid&&count){
        for(int i=threadIdx.x;i<hidden;i+=256)seed[i]=features[size_t(accepted)*hidden+i];
        if(!threadIdx.x){
            VerifyResponse result{};result.accepted=accepted;result.matched=matched;result.count=count;
            result.position=start+count;result.reason=VerifyStop(reason);result.finished=reason!=0;
            for(int i=0;i<count;++i)result.ids[i]=picks[i];
            *pending=picks[accepted];*position=result.position;*policy.remaining-=count;*response=result;
        }
    }else if(!threadIdx.x){VerifyResponse stopped{};stopped.position=start;
        stopped.reason=*invalid?VerifyStop::Invalid:VerifyStop(reason);stopped.finished=1;*response=stopped;}
}
bool aligned(const void* p){return p&&reinterpret_cast<std::uintptr_t>(p)%4==0;}
struct Span{const void* pointer;size_t bytes;bool writes;};
void separate(const Span* spans,int count){
    for(int i=0;i<count;++i){const auto start=reinterpret_cast<std::uintptr_t>(spans[i].pointer);
        if(!spans[i].bytes)continue;
        if(!aligned(spans[i].pointer)||spans[i].bytes>UINTPTR_MAX-start)
            throw std::invalid_argument("Gemma verify invalid span");
        for(int j=0;j<i;++j){if(!spans[j].bytes||(!spans[i].writes&&!spans[j].writes))continue;
            const auto other=reinterpret_cast<std::uintptr_t>(spans[j].pointer);
            if(start<other+spans[j].bytes&&other<start+spans[i].bytes)
                throw std::invalid_argument("Gemma verify overlapping lease spans");}
    }
}
bool rows_valid(int rows){return rows==2||rows==3;}
void common(int rows,int vocabulary,unsigned* invalid,cudaStream_t stream){
    if(!rows_valid(rows)||vocabulary<1||vocabulary>262144||!aligned(invalid)||!stream)
        throw std::invalid_argument("Gemma verify rows/vocabulary/device lease");
}
}
void form_verify_inputs(const int* pending,const int* first,const int* second,int* inputs,
    int rows,int vocabulary,unsigned* invalid,cudaStream_t stream){
    common(rows,vocabulary,invalid,stream);
    const Span spans[]{{pending,4,false},{first,4,false},{second,rows==3?4u:0u,false},
        {inputs,size_t(rows)*4,true},{invalid,4,true}};separate(spans,5);
    form<<<1,32,0,stream>>>(pending,first,second,inputs,rows,vocabulary,invalid);
    checked(cudaGetLastError(),"Gemma GPU verify input chain");
}
void pick_verify_rows(float* logits,float* workspace,int* picks,int rows,int vocabulary,
    float cap,unsigned* invalid,cudaStream_t stream){
    common(rows,vocabulary,invalid,stream);static_assert(sizeof(Candidate)==8);
    if(!std::isfinite(cap)||cap<=0)
        throw std::invalid_argument("Gemma verify capped row head lease");
    const int blocks=(vocabulary+255)/256;auto* partial=reinterpret_cast<Candidate*>(workspace);
    const Span spans[]{{logits,size_t(rows)*vocabulary*4,true},{workspace,size_t(rows)*blocks*8,true},
        {picks,size_t(rows)*4,true},{invalid,4,true}};separate(spans,4);
    partial_pick<<<dim3(blocks,rows),256,0,stream>>>(logits,partial,rows,vocabulary,cap,invalid);
    final_pick<<<rows,256,0,stream>>>(partial,picks,blocks,invalid);
    checked(cudaGetLastError(),"Gemma independent GPU head rows");
}
void commit_verify_rows(const int* inputs,const int* picks,const float* features,float* seed,
    int* pending,int* position,VerifyResponse* response,int rows,int hidden,int vocabulary,
    int capacity,VerifyPolicy policy,unsigned* invalid,cudaStream_t stream){
    common(rows,vocabulary,invalid,stream);
    if(hidden!=5376||capacity<rows||policy.eog_count<0||policy.eog_count>16)
        throw std::invalid_argument("Gemma verify commit/feature owner lease");
    const Span spans[]{{inputs,size_t(rows)*4,false},{picks,size_t(rows)*4,false},
        {features,size_t(rows)*hidden*4,false},{seed,size_t(hidden)*4,true},{pending,4,true},
        {position,4,true},{response,sizeof(VerifyResponse),true},{policy.remaining,4,true},
        {policy.eog_ids,size_t(policy.eog_count)*4,false},{invalid,4,true}};separate(spans,10);
    commit<<<1,256,0,stream>>>(inputs,picks,features,seed,pending,position,response,rows,
        hidden,vocabulary,capacity,policy,invalid);
    checked(cudaGetLastError(),"Gemma prefix state and logical KV commit");
}
}
