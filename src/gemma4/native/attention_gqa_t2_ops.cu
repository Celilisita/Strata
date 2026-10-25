#include "attention_gqa_t2_ops.hpp"
#include <math_constants.h>
namespace strata::gemma4::native::gqa371::detail {
namespace {
constexpr int Threads=128,Warps=Threads/32;
__device__ bool live_state(const int* position,const unsigned* invalid,int capacity){
    const long long p=*position;return p>=0&&p+1<capacity&&(!invalid||!*invalid);
}
__device__ float sum_warp(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float max_warp(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void init(Tile w){
    const int i=threadIdx.x;
    if(i<KvHeads*Rows){w.query_bad[i]=0;w.maximum[i]=0;w.denominator[i]=0;}
    if(i<KvHeads)w.first_bad_key[i]=w.capacity;
}
__global__ void unpack(Tile w,CacheView c,const int* position,const unsigned* invalid){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=std::size_t(KvHeads)*w.capacity*Dim)return;
    const int d=i%Dim,key=(i/Dim)%w.capacity,kh=i/(std::size_t(w.capacity)*Dim);
    half k=__float2half_rn(0.f),v=k;
    // Branch BEFORE a source load. Poisoned/unwritten future KV is never read.
    if(live_state(position,invalid,w.capacity)&&key<=static_cast<long long>(*position)+1){
        const auto at=((std::size_t(key)*KvHeads+kh)*(Dim/32)+d/32)*34;
        const auto* a=static_cast<const unsigned char*>(c.key)+at;
        const auto* b=static_cast<const unsigned char*>(c.value)+at;
        const float x=__half2float(*reinterpret_cast<const half*>(a))*float(reinterpret_cast<const signed char*>(a+2)[d%32]);
        const float y=__half2float(*reinterpret_cast<const half*>(b))*float(reinterpret_cast<const signed char*>(b+2)[d%32]);
        k=__float2half_rn(x);v=__float2half_rn(y);
        if(!isfinite(x)||!isfinite(y)||!isfinite(__half2float(k))||!isfinite(__half2float(v))){
            atomicMin(w.first_bad_key+kh,key);k=__float2half_rn(0.f);v=k;
        }
    }
    w.key[i]=k;w.value[i]=v;
}
__global__ void pack(Tile w,const float* query,const int* position,const unsigned* invalid){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=std::size_t(KvHeads)*Rows*Dim)return;
    const int d=i%Dim,row=i/Dim,column=row%Rows,kh=row/Rows,t=column/HeadsPerKv,h=kh*HeadsPerKv+column%HeadsPerKv;
    half q=__float2half_rn(0.f);
    if(live_state(position,invalid,w.capacity)){
        const float x=query[(std::size_t(t)*Heads+h)*Dim+d];q=__float2half_rn(x);
        if(!isfinite(x)||!isfinite(__half2float(q))){atomicOr(w.query_bad+row,1u);q=__float2half_rn(0.f);}
    }
    w.query[i]=q;
}
__global__ void normalize(Tile w,const int* position,const unsigned* invalid){
    __shared__ float maxima[Warps],sums[Warps],top;
    __shared__ unsigned errors[Warps],bad;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,column=blockIdx.x,kh=blockIdx.y,row=kh*Rows+column;
    const bool valid=live_state(position,invalid,w.capacity);const long long qpos=static_cast<long long>(*position)+column/HeadsPerKv;
    unsigned error=!valid||w.query_bad[row]||w.first_bad_key[kh]<=qpos;float maximum=-CUDART_INF_F;
    for(int j=tid;j<w.capacity;j+=Threads)if(valid&&j<=qpos){
        const float x=w.score[std::size_t(row)*w.capacity+j];error|=!isfinite(x);maximum=fmaxf(maximum,x);
    }
    maximum=max_warp(maximum);const unsigned ballot=__ballot_sync(0xffffffff,error!=0);
    if(!lane){maxima[warp]=maximum;errors[warp]=ballot;}__syncthreads();
    if(!tid){top=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
        bad=errors[0]|errors[1]|errors[2]|errors[3];if(!isfinite(top))bad=1;w.maximum[row]=bad?CUDART_NAN_F:top;}
    __syncthreads();float sum=0;
    for(int j=tid;j<w.capacity;j+=Threads){
        float p=0;
        if(!bad&&j<=qpos)p=expf(w.score[std::size_t(row)*w.capacity+j]-top);
        // Sum the actual F32 exp BEFORE RN16; never use a half/rounded denominator.
        const std::size_t at=((std::size_t(kh)*w.parts+j/PvKeys)*Rows+column)*PvKeys+j%PvKeys;
        w.probability[at]=__float2half_rn(p);sum+=p;
    }
    sum=sum_warp(sum);if(!lane)sums[warp]=sum;__syncthreads();
    if(!tid){const float denominator=(sums[0]+sums[1])+(sums[2]+sums[3]);
        if(bad||!(denominator>0)||!isfinite(denominator)){w.query_bad[row]=1;w.denominator[row]=CUDART_NAN_F;}
        else w.denominator[row]=denominator;
    }
}
__global__ void divide(Tile w,float* output,const int* position,const unsigned* invalid){
    const std::size_t i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i>=std::size_t(KvHeads)*Rows*Dim)return;
    const int d=i%Dim,row=i/Dim,column=row%Rows,kh=row/Rows,t=column/HeadsPerKv,h=kh*HeadsPerKv+column%HeadsPerKv;
    const std::size_t first=(std::size_t(kh)*w.parts*Rows+column)*Dim+d;
    float n=w.split_partial[first];
    for(int part=1;part<w.parts;++part)n+=w.split_partial[first+std::size_t(part)*Rows*Dim];
    w.partial[i]=n; // Expose the merged actual-Ph/Vh numerator to the cold oracle.
    const float denominator=w.denominator[row];
    const bool valid=live_state(position,invalid,w.capacity)&&!w.query_bad[row]&&isfinite(n)&&isfinite(denominator)&&denominator>0;
    const float value=valid?n/denominator:CUDART_NAN_F;
    output[(std::size_t(t)*Heads+h)*Dim+d]=isfinite(value)?value:CUDART_NAN_F;
}
}
void initialize(Tile w,cudaStream_t s){init<<<1,Threads,0,s>>>(w);}
void stage_kv(Tile w,CacheView c,const int* p,const unsigned* invalid,cudaStream_t s){
    unpack<<<unsigned((std::size_t(KvHeads)*w.capacity*Dim+255)/256),256,0,s>>>(w,c,p,invalid);
}
void pack_query(Tile w,const float* q,const int* p,const unsigned* invalid,cudaStream_t s){
    pack<<<(KvHeads*Rows*Dim+255)/256,256,0,s>>>(w,q,p,invalid);
}
void softmax(Tile w,const int* p,const unsigned* invalid,cudaStream_t s){normalize<<<dim3(Rows,KvHeads),Threads,0,s>>>(w,p,invalid);}
void finish(Tile w,float* output,const int* p,const unsigned* invalid,cudaStream_t s){
    divide<<<(KvHeads*Rows*Dim+255)/256,256,0,s>>>(w,output,p,invalid);
}
}
