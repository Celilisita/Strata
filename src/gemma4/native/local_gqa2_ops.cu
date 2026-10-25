#include "local_gqa2_ops.hpp"
#include <math_constants.h>
namespace strata::gemma4::native::local_gqa2_candidate::pp_tc_detail {
namespace {
__device__ float sum_warp(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float max_warp(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void pack_query(Buffers b,const float* q,float* out,int tokens,int dim,bool grouped){
    const auto i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i<std::size_t(tokens)*Heads*dim){const int d=i%dim,head=(i/dim)%Heads,token=i/(dim*Heads);
        if(grouped){
            const int first=(token/Query)*Query,queries=min(Query,tokens-first);
            b.q[(std::size_t(first)*Heads+std::size_t(head)*queries+token-first)*dim+d]=__float2half_rn(q[i]);
        }else b.q[(std::size_t(head)*tokens+token)*dim+d]=__float2half_rn(q[i]);out[i]=0;}
    if(i<std::size_t(tokens)*Heads){b.maximum[i]=-CUDART_INF_F;b.sum[i]=0;}
}
__global__ void unpack(Buffers b,CacheView c,int first,int count){
    const auto i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=std::size_t(c.heads)*count*c.dim)return;
    const int d=i%c.dim,key=(i/c.dim)%count,head=i/(c.dim*count),slot=(first+key)%c.capacity;
    const auto at=((std::size_t(slot)*c.heads+head)*(c.dim/32)+d/32)*34;
    const auto* k=static_cast<const unsigned char*>(c.key)+at;const auto* v=static_cast<const unsigned char*>(c.value)+at;
    const auto to=(std::size_t(head)*Keys+key)*c.dim+d;
    b.k[to]=__float2half_rn(__half2float(*reinterpret_cast<const half*>(k))*float(reinterpret_cast<const signed char*>(k+2)[d%32]));
    b.v[to]=__float2half_rn(__half2float(*reinterpret_cast<const half*>(v))*float(reinterpret_cast<const signed char*>(v+2)[d%32]));
}
__global__ void table(Buffers b,CacheView c,int tokens,int first,bool grouped){
    const int head=threadIdx.x;if(head>=Heads)return;
    if(grouped){
        if(head>=c.heads)return;
        const int queries=min(Query,tokens-first),columns=queries*2;
        b.kt[head]=b.k+std::size_t(head)*Keys*c.dim;b.vt[head]=b.v+std::size_t(head)*Keys*c.dim;
        b.qt[head]=b.q+(std::size_t(first)*Heads+std::size_t(head)*columns)*c.dim;
        b.pt[head]=b.probability+std::size_t(head)*columns*Keys;
        b.st[head]=b.scores+std::size_t(head)*columns*Keys;b.ot[head]=b.partial+std::size_t(head)*columns*c.dim;
        return;
    }
    const int kh=head/(Heads/c.heads);
    b.kt[head]=b.k+std::size_t(kh)*Keys*c.dim;b.vt[head]=b.v+std::size_t(kh)*Keys*c.dim;
    b.qt[head]=b.q+(std::size_t(head)*tokens+first)*c.dim;
    b.pt[head]=b.probability+std::size_t(head)*Query*Keys;
    b.st[head]=b.scores+std::size_t(head)*Query*Keys;b.ot[head]=b.partial+std::size_t(head)*Query*c.dim;
}
__global__ void softmax(Buffers b,CacheView c,const int* position,int query_first,int key_first,int count,bool grouped){
    __shared__ float sums[4],top,normalizer;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.x,q=blockIdx.y;
    const int stride=grouped?int(gridDim.y):Query;
    const int qp=*position+query_first+q,tile_row=head*stride+q;
    const auto row=std::size_t(query_first+q)*Heads+head;
    float maximum=-CUDART_INF_F;
    for(int k=tid;k<count;k+=128){const int kp=key_first+k;
        if(kp<=qp&&(!c.window||kp>=qp-c.window+1))maximum=fmaxf(maximum,b.scores[std::size_t(tile_row)*Keys+k]);}
    maximum=max_warp(maximum);if(!lane)sums[warp]=maximum;__syncthreads();
    if(!tid)top=fmaxf(fmaxf(sums[0],sums[1]),fmaxf(sums[2],sums[3]));__syncthreads();
    float sum=0;
    for(int k=tid;k<count;k+=128){const int kp=key_first+k;float p=0;
        if(kp<=qp&&(!c.window||kp>=qp-c.window+1)&&top!=-CUDART_INF_F)p=expf(b.scores[std::size_t(tile_row)*Keys+k]-top);
        b.probability[std::size_t(tile_row)*Keys+k]=__float2half_rn(p);sum+=p;}
    sum=sum_warp(sum);if(!lane)sums[warp]=sum;__syncthreads();
    if(!tid){normalizer=(sums[0]+sums[1])+(sums[2]+sums[3]);
        const float previous=b.maximum[row],old_sum=b.sum[row],next=fmaxf(previous,top);
        const float a=old_sum==0?0:expf(previous-next),v=normalizer==0?0:expf(top-next);
        b.maximum[row]=next;b.sum[row]=fmaf(old_sum,a,normalizer*v);b.a[tile_row]=a;b.b[tile_row]=v;}
}
__global__ void merge(Buffers b,float* out,int first,int queries,int dim,bool grouped){
    const auto i=std::size_t(blockIdx.x)*256+threadIdx.x;if(i>=std::size_t(Heads)*queries*dim)return;
    const int d=i%dim,q=(i/dim)%queries,head=i/(queries*dim);
    const int stride=grouped?queries:Query,tile_row=head*stride+q;
    const auto to=(std::size_t(first+q)*Heads+head)*dim+d;
    out[to]=fmaf(b.a[tile_row],out[to],b.b[tile_row]*b.partial[std::size_t(tile_row)*dim+d]);
}
__global__ void divide(Buffers b,float* out,int tokens,int dim){
    const auto i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i<std::size_t(tokens)*Heads*dim)out[i]/=b.sum[i/dim];
}
}
void prepare(Buffers b,const float* q,float* out,int t,int d,bool grouped,cudaStream_t s){pack_query<<<unsigned((std::size_t(t)*Heads*d+255)/256),256,0,s>>>(b,q,out,t,d,grouped);}
void decode(Buffers b,CacheView c,int first,int count,cudaStream_t s){unpack<<<unsigned((std::size_t(c.heads)*count*c.dim+255)/256),256,0,s>>>(b,c,first,count);}
void pointers(Buffers b,CacheView c,int t,int first,bool grouped,cudaStream_t s){table<<<1,32,0,s>>>(b,c,t,first,grouped);}
void normalize(Buffers b,CacheView c,const int* p,int q,int k,int count,int queries,bool grouped,cudaStream_t s){softmax<<<dim3(Heads,queries),128,0,s>>>(b,c,p,q,k,count,grouped);}
void accumulate(Buffers b,float* out,int first,int queries,int d,bool grouped,cudaStream_t s){merge<<<unsigned((std::size_t(Heads)*queries*d+255)/256),256,0,s>>>(b,out,first,queries,d,grouped);}
void finish(Buffers b,float* out,int t,int d,cudaStream_t s){divide<<<unsigned((std::size_t(t)*Heads*d+255)/256),256,0,s>>>(b,out,t,d);}
}
