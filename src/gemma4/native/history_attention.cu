#include "strata/gemma4/native/history.hpp"
#include <math_constants.h>
#include <algorithm>
#include <stdexcept>
namespace strata::gemma4::native {
namespace {
constexpr int Partition=128,Warps=4,MaxParts=HistoryResident/Partition;
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float maximum(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float read_q8(const unsigned char* data,size_t at,int d){
    const auto* raw=data+at+(d/32)*34;
    return __half2float(*reinterpret_cast<const __half*>(raw))*float(reinterpret_cast<const signed char*>(raw+2)[d%32]);
}
__global__ void segment(const float* query,float* partial,HistoryView v,const int* position,int row,int parts){
    __shared__ float vectors[Warps*HistoryDim],m[Warps],z[Warps];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.x,part=blockIdx.y;
    const int p=*position+row,kh=head/8,count=v.selected_count[row*HistoryKvHeads+kh];
    float* out=partial+(size_t(head)*parts+part)*(HistoryDim+2);
    if(*v.device_epoch!=v.binding.epoch||*v.invalid||p<0||p>=v.logical_capacity||p>=*v.valid_end||count<1||count>v.slots){
        for(int i=threadIdx.x;i<HistoryDim+2;i+=128)out[i]=CUDART_NAN_F;return;
    }
    float q[HistoryDim/32],a[HistoryDim/32]{};
    #pragma unroll
    for(int j=0;j<HistoryDim/32;++j)q[j]=query[size_t(head)*HistoryDim+lane+32*j];
    float top=-CUDART_INF_F,denominator=0;
    const int end=min(count*HistoryPage,(part+1)*Partition);
    const size_t selected=(size_t(row)*HistoryKvHeads+kh)*v.slots;
    for(int key=part*Partition+warp;key<end;key+=Warps){
        const int rank=key/HistoryPage,within=key%HistoryPage;
        const int page=v.selected_page[selected+rank],slot=v.selected_slot[selected+rank];
        if(page<0||page>=v.pages||slot<0||slot>=v.slots){
            if(!lane)atomicOr(v.invalid,512u);top=CUDART_NAN_F;denominator=CUDART_NAN_F;break;
        }
        const int logical=page*HistoryPage+within;
        if(logical>p||logical>=*v.valid_end)continue;
        const size_t at=(size_t(slot*HistoryPage+within)*HistoryKvHeads+kh)*HistoryRowBytes;
        float score=0;
        #pragma unroll
        for(int j=0;j<HistoryDim/32;++j)score=fmaf(q[j],read_q8(v.key,at,lane+32*j),score);
        score=sum(score);const float next=fmaxf(top,score),old=expf(top-next),current=expf(score-next);
        #pragma unroll
        for(int j=0;j<HistoryDim/32;++j)a[j]=fmaf(current,read_q8(v.value,at,lane+32*j),old*a[j]);
        denominator=fmaf(denominator,old,current);top=next;
    }
    if(!lane){m[warp]=top;z[warp]=denominator;}
    #pragma unroll
    for(int j=0;j<HistoryDim/32;++j)vectors[warp*HistoryDim+lane+32*j]=a[j];
    __syncthreads();
    const float all=fmaxf(fmaxf(m[0],m[1]),fmaxf(m[2],m[3]));float weights[Warps],total=0;
    #pragma unroll
    for(int w=0;w<Warps;++w){weights[w]=z[w]==0?0:expf(m[w]-all);total=fmaf(z[w],weights[w],total);}
    if(!threadIdx.x){out[0]=all;out[1]=total;}
    for(int d=threadIdx.x;d<HistoryDim;d+=128){float value=0;
        #pragma unroll
        for(int w=0;w<Warps;++w)value=fmaf(weights[w],vectors[w*HistoryDim+d],value);
        out[2+d]=value;}
}
__global__ void merge(const float* partial,float* output,HistoryView v,int parts){
    __shared__ float weights[MaxParts];const int lane=threadIdx.x,head=blockIdx.x;
    const float* p=partial+size_t(head)*parts*(HistoryDim+2);
    float top=-CUDART_INF_F;bool bad=*v.invalid||*v.device_epoch!=v.binding.epoch;
    for(int i=lane;i<parts;i+=32){const float z=p[i*(HistoryDim+2)+1],m=p[i*(HistoryDim+2)];
        bad|=!isfinite(z)||(z>0&&!isfinite(m));if(z>0)top=fmaxf(top,m);}
    const unsigned invalid=__ballot_sync(0xffffffff,bad);top=maximum(top);float denominator=0;
    for(int i=lane;i<parts;i+=32){const float z=p[i*(HistoryDim+2)+1];
        const float w=z==0?0:expf(p[i*(HistoryDim+2)]-top);weights[i]=w;denominator=fmaf(z,w,denominator);}
    denominator=sum(denominator);__syncwarp();
    #pragma unroll
    for(int j=0;j<HistoryDim/32;++j){float value=0;
        // Preserve174's zero-global-weight skip (not universal NaN propagation).
        for(int i=0;i<parts;++i)if(weights[i]!=0)value=fmaf(weights[i],p[i*(HistoryDim+2)+2+lane+32*j],value);
        output[size_t(head)*HistoryDim+lane+32*j]=invalid||!(denominator>0)?CUDART_NAN_F:value/denominator;}
}
bool overlap(const void* a,size_t an,const void* b,size_t bn){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<an:x-y<bn;
}
}
size_t history_attention_bytes(){return size_t(MaxParts)*HistoryHeads*(HistoryDim+2)*sizeof(float);}
void history_attention(const float* query,float* output,HistoryView v,const int* position,
    int heads,int tokens,AttentionWorkspace w,HistoryBinding live){
    history_detail::validate(v,live);const int parts=(v.resident+Partition-1)/Partition;
    const size_t bytes=size_t(parts)*HistoryHeads*(HistoryDim+2)*sizeof(float);
    if(!query||!output||!position||!w.data||heads!=HistoryHeads||tokens<1||tokens>HistoryQueries||w.bytes<bytes||
       reinterpret_cast<std::uintptr_t>(query)%4||reinterpret_cast<std::uintptr_t>(output)%4||
       reinterpret_cast<std::uintptr_t>(position)%4||reinterpret_cast<std::uintptr_t>(w.data)%4)
        throw std::invalid_argument("History Q32/D512/T1..3 workspace contract");
    const size_t io=size_t(tokens)*HistoryHeads*HistoryDim*sizeof(float);
    const void* pools[]{v.host_key,v.host_value,v.key,v.value};
    const size_t sizes[]{size_t(v.pages)*HistoryPage*HistoryKvHeads*HistoryRowBytes,
        size_t(v.pages)*HistoryPage*HistoryKvHeads*HistoryRowBytes,
        size_t(v.resident)*HistoryKvHeads*HistoryRowBytes,size_t(v.resident)*HistoryKvHeads*HistoryRowBytes};
    if(overlap(query,io,output,io)||overlap(query,io,w.data,bytes)||overlap(output,io,w.data,bytes)||
       overlap(position,4,output,io)||overlap(position,4,w.data,bytes))
        throw std::invalid_argument("History query/output/position/partial alias");
    for(int i=0;i<4;++i)if(overlap(query,io,pools[i],sizes[i])||overlap(output,io,pools[i],sizes[i])||
        overlap(w.data,bytes,pools[i],sizes[i])||overlap(position,4,pools[i],sizes[i]))
        throw std::invalid_argument("History attention aliases authoritative/resident pools");
    auto* partial=static_cast<float*>(w.data);
    // Consume a query's resident slots before the next query may replace them.
    // The selected_page diagnostic ID plans remain retained for every query.
    for(int row=0;row<tokens;++row){
        const auto* q=query+size_t(row)*HistoryHeads*HistoryDim;
        history_select(q,v,position,row,live);
        segment<<<dim3(HistoryHeads,parts),128,0,live.stream>>>(q,partial,v,position,row,parts);
        merge<<<HistoryHeads,32,0,live.stream>>>(partial,output+size_t(row)*HistoryHeads*HistoryDim,v,parts);
        primitive::check_cuda(cudaGetLastError(),"History causal selected Q8 F32 attention/merge");
    }
}
}
