#include "strata/kernels/native_gdn_cluster.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace strata::kernels {
namespace {
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for(int mask=16;mask;mask>>=1)x+=__shfl_xor_sync(0xffffffffu,x,mask);
    return x;
}
__global__ void cluster(float* __restrict__ history,const float* __restrict__ input,
    const float* __restrict__ weights,float* __restrict__ raw,float* __restrict__ activated,
    float* __restrict__ beta,const float* __restrict__ alpha,const float* __restrict__ dt,
    const float* __restrict__ a,float* __restrict__ gate,float epsilon_squared) {
    const int g=blockIdx.x,col=threadIdx.x,lane=col&31,warp=col/32;
    const int channels[4]={g*128+col,2048+g*128+col,4096+g*128+col,6144+g*128+col};
    float value[4];
#pragma unroll
    for(int j=0;j<4;++j) {
        const int c=channels[j];
        float x[4]={history[c*3],history[c*3+1],history[c*3+2],input[c]};
        float sum=0;
#pragma unroll
        for(int tap=0;tap<4;++tap)sum+=x[tap]*weights[c*4+tap];
        sum=__fadd_rn(sum,0.0f);raw[c]=sum;
        value[j]=sum/(1.0f+expf(-sum));
#pragma unroll
        for(int tap=0;tap<3;++tap)history[c*3+tap]=x[tap+1];
    }
    // Preserve current 128-thread, four-warp L2 reduction for each Q/K row.
    float qsum=warp_sum(value[0]*value[0]),ksum=warp_sum(value[1]*value[1]);
    __shared__ float sums[2][4];
    if(lane==0){sums[0][warp]=qsum;sums[1][warp]=ksum;}
    __syncthreads();
    if(col<32) {
        qsum=warp_sum(col<4?sums[0][col]:0.0f);
        ksum=warp_sum(col<4?sums[1][col]:0.0f);
        if(col==0){sums[0][0]=rsqrtf(fmaxf(qsum,epsilon_squared));sums[1][0]=rsqrtf(fmaxf(ksum,epsilon_squared));}
    }
    __syncthreads();
    activated[channels[0]]=value[0]*sums[0][0];activated[channels[1]]=value[1]*sums[1][0];
    activated[channels[2]]=value[2];activated[channels[3]]=value[3];
    if(col<2) {
        const int h=g+col*16;
        beta[h]=1.0f/(1.0f+expf(-beta[h]));
        const float x=__fadd_rn(alpha[h],dt[h]);
        gate[h]=(x>20.0f?x:log1pf(expf(x)))*a[h];
    }
}
bool valid(const void* p,size_t n) {
    const auto at=reinterpret_cast<uintptr_t>(p);return p&&at%4==0&&n<=UINTPTR_MAX-at;
}
bool overlap(const void* a,size_t an,const void* b,size_t bn) {
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+bn&&y<x+an;
}
}
NativeGdnClusterResources native_gdn_cluster_resources() noexcept {
    NativeGdnClusterResources r;cudaDeviceProp p{};cudaFuncAttributes a{};
    if(cudaGetDevice(&r.device)!=cudaSuccess||cudaGetDeviceProperties(&p,r.device)!=cudaSuccess)return r;
    r.major=p.major;r.minor=p.minor;r.sms=p.multiProcessorCount;
    if(r.major!=7||r.minor!=5||r.sms!=72)return r;
    if(cudaFuncGetAttributes(&a,cluster)!=cudaSuccess||
       cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.blocks,cluster,128,0)!=cudaSuccess)return r;
    r.regs=a.numRegs;r.local=int(a.localSizeBytes);r.shared=int(a.sharedSizeBytes);r.binary=a.binaryVersion;
    r.eligible=r.binary==75&&r.local==0&&r.shared==32&&r.regs>0&&r.regs<=255&&r.blocks>0&&a.maxThreadsPerBlock>=128;
    return r;
}
bool native_gdn_cluster_step(float* history,const float* input,const float* conv,
    float* raw,float* activated,float* beta,const float* alpha,const float* dt,
    const float* a,float* gate,float epsilon,void* stream) {
    if(!stream||!std::isfinite(epsilon)||epsilon<=0)
        throw std::invalid_argument("GDN cluster requires explicit stream/positive finite epsilon");
    const void* writable[]={history,raw,activated,beta,gate};const size_t wn[]={98304,32768,32768,128,128};
    const void* inputs[]={input,conv,alpha,dt,a};const size_t rn[]={32768,131072,128,128,128};
    for(int i=0;i<5;++i) {
        if(!valid(writable[i],wn[i])||!valid(inputs[i],rn[i]))
            throw std::invalid_argument("GDN cluster requires aligned nonnull bounded spans");
        for(int j=0;j<i;++j)if(overlap(writable[i],wn[i],writable[j],wn[j]))
            throw std::invalid_argument("GDN cluster writable alias");
        for(int j=0;j<5;++j)if(!valid(inputs[j],rn[j])||overlap(writable[i],wn[i],inputs[j],rn[j]))
            throw std::invalid_argument("GDN cluster input/output alias");
    }
    thread_local const NativeGdnClusterResources r=native_gdn_cluster_resources();
    if(!r.eligible)return false;
    cluster<<<16,128,0,static_cast<cudaStream_t>(stream)>>>(history,input,conv,raw,activated,beta,alpha,dt,a,gate,epsilon*epsilon);
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    return true;
}
}
