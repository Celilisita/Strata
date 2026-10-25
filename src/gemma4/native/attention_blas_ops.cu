#include "attention_blas_ops.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
namespace strata::gemma4::native::detail {
namespace {
__device__ float sum_warp(float v){for(int d=16;d;d/=2)v+=__shfl_xor_sync(0xffffffff,v,d);return v;}
__device__ float max_warp(float v){for(int d=16;d;d/=2)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));return v;}
__global__ void initialize(AttentionTile w,int heads,int dim){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i<size_t(heads)*QueryTile*dim)w.acc[i]=0;
    if(i<size_t(heads)*QueryTile){w.maximum[i]=-CUDART_INF_F;w.sum[i]=0;}
}
__global__ void unpack(AttentionTile w,CacheView c,int first,int count,int heads){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(heads)*KeyTile*c.dim)return;
    const int d=i%c.dim,key=(i/c.dim)%KeyTile,head=i/(KeyTile*c.dim);
    if(key>=count)return;const int kh=head/(heads/c.heads),slot=(first+key)%c.capacity;
    const size_t at=((size_t(slot)*c.heads+kh)*(c.dim/32)+d/32)*34;
    const auto* k=static_cast<const unsigned char*>(c.key)+at;const auto* v=static_cast<const unsigned char*>(c.value)+at;
    w.k[i]=__half2float(*reinterpret_cast<const half*>(k))*float(reinterpret_cast<const signed char*>(k+2)[d%32]);
    w.v[i]=__half2float(*reinterpret_cast<const half*>(v))*float(reinterpret_cast<const signed char*>(v+2)[d%32]);
}
__global__ void normalize(AttentionTile w,CacheView c,const int* position,int query_first,int key_first,int count){
    __shared__ float block[4],top,normalizer;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.x,q=blockIdx.y;
    const int qp=*position+query_first+q,row=head*QueryTile+q;
    float maximum=-CUDART_INF_F;
    for(int j=tid;j<count;j+=128){const int kp=key_first+j;
        if(kp<=qp&&(!c.window||kp>=qp-c.window+1))maximum=fmaxf(maximum,w.scores[size_t(row)*KeyTile+j]);}
    maximum=max_warp(maximum);if(!lane)block[warp]=maximum;__syncthreads();
    if(!tid)top=fmaxf(fmaxf(block[0],block[1]),fmaxf(block[2],block[3]));__syncthreads();
    float sum=0;
    for(int j=tid;j<count;j+=128){const int kp=key_first+j;float p=0;
        if(kp<=qp&&(!c.window||kp>=qp-c.window+1)&&top!=-CUDART_INF_F)p=expf(w.scores[size_t(row)*KeyTile+j]-top);
        w.scores[size_t(row)*KeyTile+j]=p;sum+=p;}
    sum=sum_warp(sum);if(!lane)block[warp]=sum;__syncthreads();
    if(!tid){normalizer=(block[0]+block[1])+(block[2]+block[3]);
        const float previous=w.maximum[row],previous_sum=w.sum[row],next=fmaxf(previous,top);
        const float a=previous_sum==0?0:expf(previous-next),b=normalizer==0?0:expf(top-next);
        w.maximum[row]=next;w.sum[row]=fmaf(previous_sum,a,normalizer*b);w.a[row]=a;w.b[row]=b;}
}
__global__ void accumulate(AttentionTile w,int heads,int queries,int dim){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(heads)*QueryTile*dim)return;
    const int row=i/dim;if(row%QueryTile>=queries)return;
    w.acc[i]=fmaf(w.a[row],w.acc[i],w.b[row]*w.partial[i]);
}
__global__ void finish(AttentionTile w,float* out,int heads,int queries,int dim,int first){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(heads)*queries*dim)return;
    const int d=i%dim,q=(i/dim)%queries,head=i/(queries*dim),row=head*QueryTile+q;
    out[(size_t(first+q)*heads+head)*dim+d]=w.acc[size_t(row)*dim+d]/w.sum[row];
}
}
void initialize_tile(AttentionTile w,int heads,int dim,cudaStream_t stream){initialize<<<unsigned((size_t(heads)*QueryTile*dim+255)/256),256,0,stream>>>(w,heads,dim);}
void unpack_tile(AttentionTile w,CacheView c,int first,int count,int heads,cudaStream_t stream){unpack<<<unsigned((size_t(heads)*KeyTile*c.dim+255)/256),256,0,stream>>>(w,c,first,count,heads);}
void normalize_tile(AttentionTile w,CacheView c,const int* position,int query_first,int key_first,int count,int queries,cudaStream_t stream){normalize<<<dim3(32,queries),128,0,stream>>>(w,c,position,query_first,key_first,count);}
void accumulate_tile(AttentionTile w,int heads,int queries,int dim,cudaStream_t stream){accumulate<<<unsigned((size_t(heads)*QueryTile*dim+255)/256),256,0,stream>>>(w,heads,queries,dim);}
void finish_tile(AttentionTile w,float* out,int heads,int queries,int dim,int first,cudaStream_t stream){finish<<<unsigned((size_t(heads)*queries*dim+255)/256),256,0,stream>>>(w,out,heads,queries,dim,first);}
}
