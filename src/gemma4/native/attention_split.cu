#include "strata/gemma4/native/attention_native.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
namespace strata::gemma4::native {
namespace {
constexpr int Partition=128,Warps=4;
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float maximum(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float read_q8(const void* data,size_t at,int i){
    const auto* p=static_cast<const unsigned char*>(data)+at+(i/32)*34;
    return __half2float(*reinterpret_cast<const half*>(p))*float(reinterpret_cast<const signed char*>(p+2)[i%32]);
}
template<int Dim>
__global__ void segment(const float* query,float* partial,CacheView c,const int* position,int heads,int parts){
    __shared__ float vectors[Warps*Dim],m[Warps],z[Warps];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.x,part=blockIdx.y,t=blockIdx.z;
    const int qpos=*position+t,first=c.window?max(0,qpos-c.window+1):0;
    if(qpos<0||(!c.window&&qpos>=c.capacity)){
        float* invalid=partial+((size_t(t)*heads+head)*parts+part)*(Dim+2);
        for(int i=threadIdx.x;i<Dim+2;i+=128)invalid[i]=CUDART_NAN_F;return;
    }
    const int begin=first+part*Partition,end=min(qpos,begin+Partition-1),kh=head/(heads/c.heads);
    float q[Dim/32],a[Dim/32]{};
    #pragma unroll
    for(int j=0;j<Dim/32;++j)q[j]=query[(size_t(t)*heads+head)*Dim+lane+32*j];
    float top=-CUDART_INF_F,denominator=0;int slot=(begin+warp)%c.capacity;
    for(int key=begin+warp;key<=end;key+=Warps){
        const size_t at=(size_t(slot)*c.heads+kh)*(Dim/32)*34;float score=0;
        #pragma unroll
        for(int j=0;j<Dim/32;++j)score=fmaf(q[j],read_q8(c.key,at,lane+32*j),score);
        score=sum(score);const float next=fmaxf(top,score),old=expf(top-next),current=expf(score-next);
        #pragma unroll
        for(int j=0;j<Dim/32;++j)a[j]=fmaf(current,read_q8(c.value,at,lane+32*j),old*a[j]);
        denominator=fmaf(denominator,old,current);top=next;slot+=Warps;if(slot>=c.capacity)slot-=c.capacity;
    }
    if(!lane){m[warp]=top;z[warp]=denominator;}
    #pragma unroll
    for(int j=0;j<Dim/32;++j)vectors[warp*Dim+lane+32*j]=a[j];
    __syncthreads();
    const float all=fmaxf(fmaxf(m[0],m[1]),fmaxf(m[2],m[3]));float weights[Warps],total=0;
    #pragma unroll
    for(int w=0;w<Warps;++w){weights[w]=z[w]==0?0:expf(m[w]-all);total=fmaf(z[w],weights[w],total);}
    float* out=partial+((size_t(t)*heads+head)*parts+part)*(Dim+2);
    if(!threadIdx.x){out[0]=all;out[1]=total;}
    for(int i=threadIdx.x;i<Dim;i+=128){float value=0;
        #pragma unroll
        for(int w=0;w<Warps;++w)value=fmaf(weights[w],vectors[w*Dim+i],value);
        out[2+i]=value;}
}
template<int Dim>
__global__ void merge(const float* partial,float* output,int heads,int parts){
    __shared__ float weights[2048];const int lane=threadIdx.x,head=blockIdx.x,t=blockIdx.y;
    const float* p=partial+(size_t(t)*heads+head)*parts*(Dim+2);
    float top=-CUDART_INF_F;bool bad=false;
    for(int i=lane;i<parts;i+=32){const float z=p[i*(Dim+2)+1],m=p[i*(Dim+2)];
        bad|=!isfinite(z)||(z>0&&!isfinite(m));if(z>0)top=fmaxf(top,m);}
    const unsigned invalid=__ballot_sync(0xffffffff,bad);top=maximum(top);float denominator=0;
    for(int i=lane;i<parts;i+=32){const float z=p[i*(Dim+2)+1];const float w=z==0?0:expf(p[i*(Dim+2)]-top);
        weights[i]=w;denominator=fmaf(z,w,denominator);}
    denominator=sum(denominator);__syncwarp();
    #pragma unroll
    for(int j=0;j<Dim/32;++j){float value=0;
        for(int i=0;i<parts;++i)if(weights[i]!=0)value=fmaf(weights[i],p[i*(Dim+2)+2+lane+32*j],value);
        output[(size_t(t)*heads+head)*Dim+lane+32*j]=invalid||!(denominator>0)?CUDART_NAN_F:value/denominator;}
}
}
void split_attention(const float* q,float* output,CacheView c,const int* position,int heads,int tokens,AttentionWorkspace w,cudaStream_t stream){
    if(!q||!output||q==output||!position||!c.key||!c.value||c.heads<1||heads<1||heads%c.heads||c.capacity<1||tokens<1||tokens>3||tokens>c.capacity||
        c.window<0||(c.dim!=256&&c.dim!=512)||!w.data)throw std::invalid_argument("Gemma split attention contract");
    const int length=c.window?(std::min)(c.window,c.capacity):c.capacity,parts=(length+Partition-1)/Partition;
    if(parts>2048||size_t(parts)*heads*tokens*(c.dim+2)*sizeof(float)>w.bytes)throw std::invalid_argument("Gemma split attention arena");
    auto* p=static_cast<float*>(w.data);
    if(c.dim==256){segment<256><<<dim3(heads,parts,tokens),128,0,stream>>>(q,p,c,position,heads,parts);merge<256><<<dim3(heads,tokens),32,0,stream>>>(p,output,heads,parts);}
    else{segment<512><<<dim3(heads,parts,tokens),128,0,stream>>>(q,p,c,position,heads,parts);merge<512><<<dim3(heads,tokens),32,0,stream>>>(p,output,heads,parts);}
    primitive::check_cuda(cudaGetLastError(),"Gemma packedQ8 parallel split attention");
}
}
