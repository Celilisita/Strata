#include "strata/gemma4/native/decoder_ops.hpp"
#include <cuda_fp16.h>
#include <cmath>
namespace strata::gemma4::native {
namespace {
__device__ float sum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void norm_rope(const float* x,float* y,const float* w,int dim,int heads,int tokens,float epsilon,float base,
    const float* frequencies,const int* position,bool rotate){
    __shared__ float partial[4],inverse;
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;const size_t first=size_t(row)*dim;
    float total=0;for(int k=tid;k<dim;k+=128)total=__fmaf_rn(x[first+k],x[first+k],total);
    total=sum(total);if(!lane)partial[warp]=total;__syncthreads();
    if(!tid)inverse=rsqrtf(((partial[0]+partial[1])+(partial[2]+partial[3]))/dim+epsilon);__syncthreads();
    if(rotate){
        for(int pair=tid;pair<dim/2;pair+=128){
            float a=x[first+pair]*inverse,b=x[first+pair+dim/2]*inverse;
            if(w){a*=w[pair];b*=w[pair+dim/2];}
            double theta=double(*position+row/heads)*pow(double(base),-2.0*pair/dim);
            if(frequencies)theta/=double(frequencies[pair]);
            double sine,cosine;sincos(theta,&sine,&cosine);
            const float sn=float(sine),cs=float(cosine);
            y[first+pair]=__fmaf_rn(-b,sn,a*cs);y[first+pair+dim/2]=__fmaf_rn(a,sn,b*cs);
        }
    }else for(int k=tid;k<dim;k+=128)y[first+k]=x[first+k]*inverse*(w?w[k]:1.f);
}
__global__ void pack(const float* key,const float* value,CacheView c,const int* position,int tokens,unsigned* invalid){
    const int lane=threadIdx.x,group=blockIdx.x%(c.dim/32),head=blockIdx.x/(c.dim/32),t=blockIdx.y;
    const size_t source=(size_t(t)*c.heads+head)*c.dim+group*32+lane;
    const int absolute=*position+t,slot=absolute%c.capacity;
    const size_t offset=((size_t(slot)*c.heads+head)*(c.dim/32)+group)*34;
    for(int which=0;which<2;++which){
        const float x=which?value[source]:key[source];float maximum=fabsf(x);
        for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
        const float scale=maximum/127.f;
        if(!isfinite(x))atomicExch(invalid,1u);
        auto* out=static_cast<unsigned char*>(which?c.value:c.key)+offset;
        if(!lane)*reinterpret_cast<half*>(out)=__float2half_rn(scale);
        reinterpret_cast<signed char*>(out+2)[lane]=static_cast<signed char>(maximum>0?max(-127,min(127,__float2int_rn(x/scale))):0);
    }
}
__device__ float cached(const void* data,size_t first,int index){
    const auto* block=static_cast<const unsigned char*>(data)+first+(index/32)*34;
    return __half2float(*reinterpret_cast<const half*>(block))*float(reinterpret_cast<const signed char*>(block+2)[index%32]);
}
__global__ void attend(const float* query,float* out,CacheView c,const int* position,int heads,int tokens){
    const int lane=threadIdx.x,head=blockIdx.x,t=blockIdx.y,dim=c.dim;
    const int qpos=*position+t,kvhead=head/(heads/c.heads),first=c.window?max(0,qpos-c.window+1):0;
    float q[16],acc[16]{};for(int j=0;j<dim/32;++j)q[j]=query[(size_t(t)*heads+head)*dim+lane+32*j];
    float maximum=-INFINITY,normalizer=0;int slot=first%c.capacity;
    for(int keypos=first;keypos<=qpos;++keypos){
        const size_t at=(size_t(slot)*c.heads+kvhead)*(dim/32)*34;float score=0;
        for(int j=0;j<dim/32;++j)score=__fmaf_rn(q[j],cached(c.key,at,lane+32*j),score);
        score=sum(score);const float next=fmaxf(maximum,score),a=__expf(maximum-next),b=__expf(score-next);
        for(int j=0;j<dim/32;++j)acc[j]=__fmaf_rn(b,cached(c.value,at,lane+32*j),a*acc[j]);
        normalizer=__fmaf_rn(normalizer,a,b);maximum=next;if(++slot==c.capacity)slot=0;
    }
    for(int j=0;j<dim/32;++j)out[(size_t(t)*heads+head)*dim+lane+32*j]=acc[j]/normalizer;
}
}
void head_norm_rope(const float* x,float* y,const float* weight,int dim,int heads,int tokens,float epsilon,float base,
    const float* frequencies,const int* position,bool rotate,cudaStream_t stream){
    if(!x||!y||!position||(dim!=256&&dim!=512)||heads<1||tokens<1||epsilon<=0||(rotate&&base<=0))throw std::invalid_argument("Gemma head norm/RoPE shape");
    norm_rope<<<heads*tokens,128,0,stream>>>(x,y,weight,dim,heads,tokens,epsilon,base,frequencies,position,rotate);
    primitive::check_cuda(cudaGetLastError(),"Gemma weighted QK/unweighted V norm and RoPE");
}
void store_kv(const float* k,const float* v,CacheView c,const int* position,int tokens,unsigned* invalid,cudaStream_t stream){
    if(!k||!v||!c.key||!c.value||c.key==c.value||!position||!invalid||c.capacity<tokens||c.heads<1||(c.dim!=256&&c.dim!=512))throw std::invalid_argument("Gemma Q8 cache lease");
    pack<<<dim3(c.heads*(c.dim/32),tokens),32,0,stream>>>(k,v,c,position,tokens,invalid);primitive::check_cuda(cudaGetLastError(),"Gemma Q8 KV append");
}
void attention(const float* q,float* out,CacheView c,const int* position,int heads,int tokens,cudaStream_t stream){
    if(!q||!out||q==out||!c.key||!c.value||!position||c.capacity<1||c.heads<1||heads<1||heads%c.heads||tokens<1||tokens>c.capacity||
       (c.dim!=256&&c.dim!=512)||c.window<0)throw std::invalid_argument("Gemma attention geometry");
    attend<<<dim3(heads,tokens),32,0,stream>>>(q,out,c,position,heads,tokens);primitive::check_cuda(cudaGetLastError(),"Gemma causal GQA/SWA scale-one attention");
}
}
