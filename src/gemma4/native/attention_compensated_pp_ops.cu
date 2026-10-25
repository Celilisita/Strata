#include "attention_compensated_pp_ops.hpp"
#include <math_constants.h>

namespace strata::gemma4::native::detail286 {
namespace {
__device__ void pair(float x,half* high,half* low){
    const half h=__float2half_rn(x);*high=h;
    *low=__float2half_rn((x-__half2float(h))*4096.f);
}
__device__ float sum_warp(float x){
    for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;
}
__device__ float max_warp(float x){
    for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;
}
__global__ void unpack(PhysicalKv w,CacheView c){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    const std::size_t total=std::size_t(KvHeads)*w.end*Dim;
    if(i>=total)return;
    const int d=i%Dim,key=(i/Dim)%w.end,head=i/(std::size_t(w.end)*Dim);
    const auto at=((std::size_t(key)*KvHeads+head)*16+d/32)*34;
    const auto* k=static_cast<const unsigned char*>(c.key)+at;
    const auto* v=static_cast<const unsigned char*>(c.value)+at;
    const float kval=__half2float(*reinterpret_cast<const half*>(k))*
        float(reinterpret_cast<const signed char*>(k+2)[d%32]);
    const float vval=__half2float(*reinterpret_cast<const half*>(v))*
        float(reinterpret_cast<const signed char*>(v+2)[d%32]);
    pair(kval,w.k+i,w.k+total+i);
    half* bank=head<2?w.v01:w.v23;
    const auto vi=(std::size_t(head%2)*w.end+key)*Dim+d;
    pair(vval,bank+vi,bank+total/2+vi);
}
__global__ void query(AttentionTile w,const float* q,int first){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=TileElements)return;
    const int d=i%Dim,qid=(i/Dim)%QueryTile,head=i/(QueryTile*Dim);
    pair(q[(std::size_t(first+qid)*Heads+head)*Dim+d],w.q+i,w.q+TileElements+i);
}
__global__ void pointers(PointerTables t,PhysicalKv kv,AttentionTile w,int key){
    const int kh=threadIdx.x;if(kh>=KvHeads)return;
    const auto tile=std::size_t(kh)*HeadsPerKv*QueryTile*Dim;
    const auto ki=(std::size_t(kh)*kv.end+key)*Dim;
    const auto vi=(std::size_t(kh%2)*kv.end+key)*Dim;
    half* value=kh<2?kv.v01:kv.v23;
    const auto kplane=std::size_t(KvHeads)*kv.end*Dim;
    for(int plane=0;plane<2;++plane){
        t.key[plane][kh]=kv.k+ki+plane*kplane;
        t.value[plane][kh]=value+vi+plane*kplane/2;
        t.query[plane][kh]=w.q+tile+plane*TileElements;
        t.probability[plane][kh]=w.p+tile+plane*TileElements;
    }
    // KeyTile==Dim: scores and PV partials have identical group byte strides.
    t.work[kh]=w.work+tile;
}
__global__ void initialize(AttentionTile w){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i<TileElements)w.acc[i]=0;
    if(i<std::size_t(Heads)*QueryTile){w.maximum[i]=-CUDART_INF_F;w.sum[i]=0;}
}
__global__ void normalize(AttentionTile w,const int* position,int past,int first,int key){
    __shared__ float block[4],top,normalizer;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int head=blockIdx.x,q=blockIdx.y,row=head*QueryTile+q;
    // A stale GPU cursor is an invalid request. It cannot silently use the
    // host geometry with a different causal mask; final output becomesNaN.
    const bool valid=*position==past;const int qp=past+first+q;
    float maximum=-CUDART_INF_F;
    for(int j=tid;j<KeyTile;j+=128)
        if(valid&&key+j<=qp)maximum=fmaxf(maximum,w.work[std::size_t(row)*KeyTile+j]);
    maximum=max_warp(maximum);if(!lane)block[warp]=maximum;__syncthreads();
    if(!tid)top=fmaxf(fmaxf(block[0],block[1]),fmaxf(block[2],block[3]));__syncthreads();
    float sum=0;
    for(int j=tid;j<KeyTile;j+=128){
        float p=valid?0.f:CUDART_NAN_F;
        if(valid&&key+j<=qp&&top!=-CUDART_INF_F)
            p=expf(w.work[std::size_t(row)*KeyTile+j]-top);
        const auto at=std::size_t(row)*KeyTile+j;
        pair(p,w.p+at,w.p+TileElements+at);sum+=p;
    }
    sum=sum_warp(sum);if(!lane)block[warp]=sum;__syncthreads();
    if(!tid){
        normalizer=(block[0]+block[1])+(block[2]+block[3]);
        const float previous=w.maximum[row],previous_sum=w.sum[row],next=fmaxf(previous,top);
        const float a=previous_sum==0?0:expf(previous-next);
        const float b=normalizer==0?0:expf(top-next);
        w.maximum[row]=next;w.sum[row]=fmaf(previous_sum,a,normalizer*b);
        w.a[row]=a;w.b[row]=b;
    }
}
__global__ void accumulate(AttentionTile w){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=TileElements)return;const int row=i/Dim;
    w.acc[i]=fmaf(w.a[row],w.acc[i],w.b[row]*w.work[i]);
}
__global__ void finish(AttentionTile w,float* output,int first){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=TileElements)return;
    const int d=i%Dim,q=(i/Dim)%QueryTile,head=i/(QueryTile*Dim),row=head*QueryTile+q;
    output[(std::size_t(first+q)*Heads+head)*Dim+d]=w.acc[i]/w.sum[row];
}
}
void unpack_physical(PhysicalKv w,CacheView c,cudaStream_t s){
    unpack<<<unsigned((std::size_t(KvHeads)*w.end*Dim+255)/256),256,0,s>>>(w,c);
}
void pack_query(AttentionTile w,const float* q,int first,cudaStream_t s){
    query<<<unsigned((TileElements+255)/256),256,0,s>>>(w,q,first);
}
void prepare_pointers(PointerTables t,PhysicalKv kv,AttentionTile w,int key,cudaStream_t s){
    pointers<<<1,KvHeads,0,s>>>(t,kv,w,key);
}
void initialize_tile(AttentionTile w,cudaStream_t s){
    initialize<<<unsigned((TileElements+255)/256),256,0,s>>>(w);
}
void normalize_tile(AttentionTile w,const int* p,int past,int first,int key,cudaStream_t s){
    normalize<<<dim3(Heads,QueryTile),128,0,s>>>(w,p,past,first,key);
}
void accumulate_tile(AttentionTile w,cudaStream_t s){
    accumulate<<<unsigned((TileElements+255)/256),256,0,s>>>(w);
}
void finish_tile(AttentionTile w,float* out,int first,cudaStream_t s){
    finish<<<unsigned((TileElements+255)/256),256,0,s>>>(w,out,first);
}
}
