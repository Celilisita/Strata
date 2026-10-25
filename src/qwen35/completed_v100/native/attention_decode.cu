#include "strata/qwen35/completed_v100/native/decode_graph_patch.hpp"
#include "strata/qwen35/completed_v100/native/attention_decode.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace strata::completed_v100::qwen35::native {
namespace {
__device__ float warp_sum(float value) {
    for(int d=16;d;d/=2) value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,d));
    return value;
}
__device__ float warp_max(float value) {
    for(int d=16;d;d/=2) value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,d));
    return value;
}
template<int Tile,bool Dynamic=false,bool Q8=false> __global__ void segments(const float *q, const unsigned char *k, const unsigned char *v,
                                          float *parts,int heads,int kvheads,int valid,int count,float scale,const DecodeCursor *cursor=nullptr) {
    if constexpr(Dynamic){valid=cursor->position+1;count=cursor->partitions;if(blockIdx.y>=unsigned(count))return;}
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
            const auto *key=k+(size_t(first+row)*kvheads+kh)*(Q8?272:144);
            #pragma unroll
            for(int word=lane;word<64;word+=32) {
                const int block=word/8, group=word%8;
                const auto *encoded=key+block*(Q8?34:18);
                unsigned packed;memcpy(&packed,encoded+2+(Q8?group:group%4)*4,4);
                if constexpr(!Q8)packed=(packed>>(group<4?0:4))&0x0f0f0f0f;
                const int integer=__dp4a(int(packed),query[word],0);
                const float adjusted=Q8?__fmul_rn(float(integer),delta[block]):
                    __fmaf_rn(float(integer),delta[block],-original_sum[block]);
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
                        const auto *value=v+(size_t(first+row)*kvheads+kh)*(Q8?272:144)+(d/32)*(Q8?34:18);
                        const int code=Q8?int(reinterpret_cast<const signed char *>(value+2)[d%32]):
                            int((value[2+d%16]>>(d%32<16?0:4))&15)-8;
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
template<bool Dynamic=false>
__global__ void combine(const float *parts,float *out,int count,const DecodeCursor *cursor=nullptr) {
    if constexpr(Dynamic)count=cursor->partitions;
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
void validate(DecodeAttentionShape s) {
    if(s.dim!=256 || s.query_heads<1 || s.query_heads>256 || s.kv_heads<1 || s.query_heads%s.kv_heads ||
       s.valid<1 || s.valid>s.capacity || s.capacity>262144 || (s.kv_tile!=64 && s.kv_tile!=128 && s.kv_tile!=256) ||
       s.partitions<0 || s.partitions>(s.valid+s.kv_tile-1)/s.kv_tile ||
       !(s.scale>0) || !std::isfinite(s.scale))throw std::invalid_argument("invalid own decode attention geometry");
}
}
int decode_partition_count(int valid,int heads,DecodeRounding r){
    if(valid<1 || valid>262144 || heads<1 || heads>256 || r.processors<1 || r.processors>4096 ||
       r.resident_blocks<1 || r.resident_blocks>32 || r.key_granularity!=256)
        throw std::invalid_argument("invalid decode partition rounding policy");
    return decode_partitions_unchecked(valid,heads,r);
}
size_t decode_attention_workspace(DecodeAttentionShape s) {
    validate(s);return size_t(s.query_heads)*(s.partitions?s.partitions:(s.valid+s.kv_tile-1)/s.kv_tile)*258*4;
}
void decode_attention(const float *q,const void *k,const void *v,float *out,void *work,size_t bytes,DecodeAttentionShape s,cudaStream_t stream) {
    const size_t required=decode_attention_workspace(s);
    if(!q || !k || !v || !out || !work || bytes<required)throw std::invalid_argument("own decode attention buffer extent mismatch");
    const int count=s.partitions?s.partitions:(s.valid+s.kv_tile-1)/s.kv_tile;
    const dim3 grid(s.query_heads,count);
    auto *part=static_cast<float *>(work);const auto *key=static_cast<const unsigned char *>(k),*value=static_cast<const unsigned char *>(v);
    if(s.q8){
        if(s.kv_tile==64)segments<64,false,true><<<grid,128,0,stream>>>(q,key,value,part,s.query_heads,s.kv_heads,s.valid,count,s.scale);
        else if(s.kv_tile==128)segments<128,false,true><<<grid,128,0,stream>>>(q,key,value,part,s.query_heads,s.kv_heads,s.valid,count,s.scale);
        else segments<256,false,true><<<grid,128,0,stream>>>(q,key,value,part,s.query_heads,s.kv_heads,s.valid,count,s.scale);
    }else if(s.kv_tile==64)segments<64><<<grid,128,0,stream>>>(q,key,value,part,s.query_heads,s.kv_heads,s.valid,count,s.scale);
    else if(s.kv_tile==128)segments<128><<<grid,128,0,stream>>>(q,key,value,part,s.query_heads,s.kv_heads,s.valid,count,s.scale);
    else segments<256><<<grid,128,0,stream>>>(q,key,value,part,s.query_heads,s.kv_heads,s.valid,count,s.scale);
    check_cuda(cudaGetLastError(),"own split-KV decode attention");
    combine<false><<<s.query_heads,256,0,stream>>>(part,out,count);
    check_cuda(cudaGetLastError(),"own split-KV decode combination");
}
DecodeNode decode_attention_node(void *function){
    if(function==reinterpret_cast<void *>(segments<128>)||function==reinterpret_cast<void *>(segments<128,true>)||
       function==reinterpret_cast<void *>(segments<128,false,true>)||function==reinterpret_cast<void *>(segments<128,true,true>))return DecodeNode::Segments;
    return (function==reinterpret_cast<void *>(combine<false>)||function==reinterpret_cast<void *>(combine<true>))?DecodeNode::Combine:DecodeNode::None;
}
void update_decode_attention(cudaGraphExec_t exec,cudaGraphNode_t node,const cudaKernelNodeParams &original,DecodeNode kind,int position,int heads,DecodeRounding rounding){
    int valid=position+1,count=decode_partition_count(valid,heads,rounding);auto p=original;
    if(kind==DecodeNode::Segments){
        auto args=borrowed_graph_arguments<10>(original);
        if(*static_cast<const int *>(args[4])!=heads)throw std::invalid_argument("graph attention heads changed");
        args[6]=&valid;args[7]=&count;p.kernelParams=args.data();p.gridDim.y=count;
        check_cuda(cudaGraphExecKernelNodeSetParams(exec,node,&p),"update own graph actual KV length/grid");
    }else if(kind==DecodeNode::Combine){
        auto args=borrowed_graph_arguments<4>(original);args[2]=&count;p.kernelParams=args.data();
        check_cuda(cudaGraphExecKernelNodeSetParams(exec,node,&p),"update own graph actual KV reduction length");
    }else throw std::invalid_argument("invalid graph attention node");
}

void decode_attention_graph(const float *q,const void *k,const void *v,float *out,void *work,size_t bytes,
                            DecodeAttentionShape shape,const DecodeCursor *cursor,int maximum,cudaStream_t stream){
    validate(shape);
    if(!q||!k||!v||!out||!work||!cursor||shape.kv_tile!=128||maximum<1||maximum>1024||bytes<size_t(shape.query_heads)*maximum*258*4)
        throw std::invalid_argument("invalid bounded device attention workspace");
    auto *parts=static_cast<float *>(work);
    if(shape.q8)segments<128,true,true><<<dim3(shape.query_heads,maximum),128,0,stream>>>(q,static_cast<const unsigned char *>(k),static_cast<const unsigned char *>(v),parts,shape.query_heads,shape.kv_heads,shape.valid,shape.partitions,shape.scale,cursor);
    else segments<128,true><<<dim3(shape.query_heads,maximum),128,0,stream>>>(q,static_cast<const unsigned char *>(k),static_cast<const unsigned char *>(v),parts,shape.query_heads,shape.kv_heads,shape.valid,shape.partitions,shape.scale,cursor);
    check_cuda(cudaGetLastError(),"own graph dynamic exact active KV partitions");
    combine<true><<<shape.query_heads,256,0,stream>>>(parts,out,shape.partitions,cursor);
    check_cuda(cudaGetLastError(),"own graph dynamic exact KV reduction");
}

}
