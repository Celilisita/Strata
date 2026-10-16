#include "strata/qwen35/native/attention_ram.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstring>
namespace strata::qwen35::native { namespace {
__device__ float warp_sum(float value) {
    for(int d=16;d;d/=2) value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,d));
    return value;
}
__device__ float warp_max(float value) {
    for(int d=16;d;d/=2) value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,d));
    return value;
}
template<int Tile> __global__ void segments(const float *q, const unsigned char *k, const unsigned char *v,
                                          float *parts,int heads,int kvheads,int valid,int count,float scale,const int *positions,int shift,int capacity,int window,DecodeRounding rounding,RamKvView ram) {
    valid=positions[0]-shift+1;
    const int logical_valid=valid;valid=min((valid+RamPage-1)/RamPage,ram.selected_slots)*RamPage;
    count=decode_partitions_unchecked(valid,heads,rounding);if(blockIdx.y>=unsigned(count))return;
    constexpr int Groups=Tile<128?Tile/32:4,Steps=(Tile+127)/128;
    __shared__ int query[64];
    __shared__ float delta[8], original_sum[8], score[Tile], probability[Tile];
    __shared__ float local_max[Groups], local_scale[Groups], history_scale[Groups], reduction[4], maximum, denominator;
    const int lane=threadIdx.x%32, warp=threadIdx.x/32, head=blockIdx.x;
    const int kh=head/(heads/kvheads);
    // Four-value serial sums followed by an eight-worker butterfly retain the
    // F32 affine correction independently of the packed Q8 integer plane.
    if(threadIdx.x<64){
        const int block=threadIdx.x/8, worker=threadIdx.x%8;
        float values[4], largest=0, sum=0;
        for(int j=0;j<4;++j){
            values[j]=__fmul_rn(q[head*256+block*32+worker*4+j],scale);
            largest=fmaxf(largest,fabsf(values[j]));sum=__fadd_rn(sum,values[j]);
        }
        for(int step=4;step;step/=2){
            largest=fmaxf(largest,__shfl_xor_sync(0xffffffff,largest,step,8));
            sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,step,8));
        }
        const float d=__fmul_rn(largest,1.0f/127.0f);unsigned packed=0;
        for(int j=0;j<4;++j){const int code=d==0?0:int(roundf(__fdividef(values[j],d)));packed|=(unsigned(code)&255)<<(8*j);}
        query[block*8+worker]=int(packed);
        if(!worker){delta[block]=d;original_sum[block]=sum;}
    }
    __syncthreads();
    float sum=0,current=-CUDART_INF_F,accum[Groups][2]{};
    // Each partition owns a strided sequence of tiles. Live storage is bounded
    // by one tile even when the context spans many online updates.
    for(int first=blockIdx.y*Tile;first<valid;first+=count*Tile){
        const int extent=min(Tile,valid-first);
    for(int row=warp;row<Tile;row+=4) {
        float dot=0;
        if(row<extent) {
            const int rank=(first+row)/RamPage,inpage=(first+row)%RamPage;
            const int logical=ram.sorted_ids[size_t(kh)*ram.pages+rank]*RamPage+inpage;
            const int physical=ram.selected_slot[size_t(kh)*ram.slots+rank]*RamPage+inpage;
            const auto *key=k+(size_t(physical)*kvheads+kh)*272;
            if(logical>=logical_valid)dot=-CUDART_INF_F;
            else
            #pragma unroll
            for(int word=lane;word<64;word+=32) {
                const int block=word/8, group=word%8;
                const auto *encoded=key+block*34;
                unsigned packed;memcpy(&packed,encoded+2+group*4,4);
                
                const int integer=__dp4a(int(packed),query[word],0);
                const float adjusted=__fmul_rn(float(integer),delta[block]);
                dot=__fmaf_rn(__half2float(*reinterpret_cast<const __half *>(encoded)),adjusted,dot);
            }
        }
        dot=warp_sum(dot);
        if(!lane)score[row]=row<extent?dot:-CUDART_INF_F;
    }
    __syncthreads();
        for(int step=0;step<Steps;++step){
            if(warp<Groups){
                const int row=step*128+warp*32+lane;
                const float fresh=__fadd_rn(warp_max(score[row]),3.0f*0.6931f);
                const float next=fmaxf(current,fresh);
                const float history=isinf(next)?1.0f:__expf(__fsub_rn(current,next));
                const float p=row<extent?__expf(__fsub_rn(score[row],next)):0.0f;
                probability[row]=p;sum=__fmaf_rn(sum,history,p);current=next;
                if(!lane)history_scale[warp]=history;
            }
            __syncthreads();
            #pragma unroll
            for(int half=0;half<2;++half){
                const int d=threadIdx.x+half*128;
                #pragma unroll
                for(int group=0;group<Groups;++group){
                    float partial=accum[group][half];const int begin=step*128+group*32,end=min(begin+32,extent);
                    const float history=history_scale[group];
                    if(begin>=end)partial=__fmul_rn(partial,history);
                    for(int row=begin;row<end;++row){
                        const int rank=(first+row)/RamPage,inpage=(first+row)%RamPage;
                        const int logical=ram.sorted_ids[size_t(kh)*ram.pages+rank]*RamPage+inpage;
                        if(logical>=logical_valid){if(row==begin)partial=__fmul_rn(partial,history);continue;}
                        const int physical=ram.selected_slot[size_t(kh)*ram.slots+rank]*RamPage+inpage;
                        const auto *value=v+(size_t(physical)*kvheads+kh)*272+(d/32)*34;
                        const int code=int(reinterpret_cast<const signed char *>(value+2)[d%32]);
                        const float decoded=__fmul_rn(__half2float(*reinterpret_cast<const __half *>(value)),float(code));
                        partial=row==begin?__fmaf_rn(history,partial,__fmul_rn(probability[row],decoded)):
                                           __fmaf_rn(probability[row],decoded,partial);
                    }
                    accum[group][half]=partial;
                }
            }
            __syncthreads();
        }
    }
    if(warp<Groups && !lane)local_max[warp]=current;
    __syncthreads();
    if(!threadIdx.x){
        float m=-CUDART_INF_F;for(int group=0;group<Groups;++group)m=fmaxf(m,local_max[group]);maximum=m;
        for(int group=0;group<Groups;++group)local_scale[group]=__expf(__fsub_rn(local_max[group],m));
    }
    __syncthreads();
    sum=warp_sum(warp<Groups?__fmul_rn(sum,local_scale[warp]):0.0f);
    if(!lane)reduction[warp]=sum;
    __syncthreads();
    if(!threadIdx.x)denominator=__fadd_rn(__fadd_rn(reduction[0],reduction[2]),__fadd_rn(reduction[1],reduction[3]));
    __syncthreads();
    const size_t offset=(size_t(head)*count+blockIdx.y)*258;
    for(int half=0;half<2;++half){
        float result=0;
        for(int group=0;group<Groups;++group)result=__fadd_rn(result,__fmul_rn(accum[group][half],local_scale[group]));
        parts[offset+threadIdx.x+half*128]=result;
    }
    if(!threadIdx.x){parts[offset+256]=maximum;parts[offset+257]=denominator;}
}
__global__ void combine(const float *parts,float *out,int count,const int *positions,int shift,int window,int heads,DecodeRounding rounding) {
    count=decode_partitions_unchecked(min(positions[0]-shift+1,window),heads,rounding);
    __shared__ float maximum;
    const int head=blockIdx.x,d=threadIdx.x;
    if(!d) {
        float m=-CUDART_INF_F;
        for(int p=0;p<count;++p)m=fmaxf(m,parts[(size_t(head)*count+p)*258+256]);
        maximum=m;
    }
    __syncthreads();
    float numerator=0,denominator=0;
    for(int p=0;p<count;++p) {
        const size_t offset=(size_t(head)*count+p)*258;
        const float scale=__expf(__fsub_rn(parts[offset+256],maximum));
        numerator=__fmaf_rn(scale,parts[offset+d],numerator);
        denominator=__fmaf_rn(scale,parts[offset+257],denominator);
    }
    out[size_t(head)*256+d]=__fdividef(numerator,denominator);
}
}
DecodeNode ram_decode_node(void *function){
    if(function==reinterpret_cast<void *>(segments<128>))return DecodeNode::Segments;
    return function==reinterpret_cast<void *>(combine)?DecodeNode::Combine:DecodeNode::None;
}
void ram_attention(const float *q,RamKvView ram,float *out,void *work,size_t bytes,const int *positions,
                   int shift,int tokens,int heads,DecodeRounding rounding,cudaStream_t stream){
    const int maximum=decode_partition_count(ram.selected_tokens,heads,rounding);
    if(!q||!out||!work||!positions||tokens<1||tokens>3||bytes<size_t(heads)*maximum*258*4)
        throw std::invalid_argument("invalid RAM selected decode workspace");
    for(int token=0;token<tokens;++token){
        const float *query=q+size_t(token)*heads*256;float *output=out+size_t(token)*heads*256;
        ram_select(query,ram,positions+token,shift,heads,stream);
        if(tokens>1){ram_verify_one(query,ram,output,work,bytes,positions+token,shift,heads,stream);continue;}
        segments<128><<<dim3(heads,maximum),128,0,stream>>>(query,ram.key,ram.value,static_cast<float *>(work),heads,ram.heads,
            ram.selected_tokens,maximum,1.f/16.f,positions+token,shift,ram.resident,ram.selected_tokens,rounding,ram);
        check_cuda(cudaGetLastError(),"RAM selected direct Q8 decode");
        combine<<<heads,256,0,stream>>>(static_cast<const float *>(work),output,maximum,positions+token,shift,ram.selected_tokens,heads,rounding);
        check_cuda(cudaGetLastError(),"RAM selected decode merge");
    }
}
}
