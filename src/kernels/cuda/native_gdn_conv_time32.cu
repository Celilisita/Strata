#include "strata/kernels/native_gdn_conv_time32.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
namespace {
constexpr int Channels=8192,Time=32,Threads=256;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void time32(const float* __restrict__ history,const float* __restrict__ input,
                       const float* __restrict__ weights,float* __restrict__ output,int tokens) {
    const int c=int(blockIdx.x)*Threads+threadIdx.x,first=int(blockIdx.y)*Time;
    float values[4],w[4];
#pragma unroll
    for(int tap=0;tap<3;++tap){const int t=first-3+tap;
        values[tap]=t<0?history[c*3+tap]:input[size_t(t)*Channels+c];}
#pragma unroll
    for(int tap=0;tap<4;++tap)w[tap]=weights[c*4+tap];
    const int end=first+Time<tokens?first+Time:tokens;
    for(int t=first;t<end;++t){
        values[3]=input[size_t(t)*Channels+c];float sum=0.0f;
#pragma unroll
        for(int tap=0;tap<4;++tap)sum+=values[tap]*w[tap];
        sum=__fadd_rn(sum,0.0f);
        output[size_t(t)*Channels+c]=sum/(1.0f+expf(-sum));
#pragma unroll
        for(int tap=0;tap<3;++tap)values[tap]=values[tap+1];
    }
}
__global__ void commit_history(float* __restrict__ history,const float* __restrict__ input,int tokens){
    const int c=int(blockIdx.x)*Threads+threadIdx.x;
#pragma unroll
    for(int tap=0;tap<3;++tap)history[c*3+tap]=input[size_t(tokens-3+tap)*Channels+c];
}
bool span(const void* p,size_t bytes){const auto n=reinterpret_cast<uintptr_t>(p);
    return p&&n%sizeof(float)==0&&bytes<=UINTPTR_MAX-n;}
bool overlap(const void* a,size_t an,const void* b,size_t bn){const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<y+bn&&y<x+an;}
}
NativeGdnConvTime32Resources native_gdn_conv_time32_resources(){
    NativeGdnConvTime32Resources r;int device=0;cudaDeviceProp p{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&p,device));
    r.major=p.major;r.minor=p.minor;r.sms=p.multiProcessorCount;if(r.major!=7||r.minor!=5)return r;
    cudaFuncAttributes a{},b{};check(cudaFuncGetAttributes(&a,time32));check(cudaFuncGetAttributes(&b,commit_history));
    r.conv_regs=a.numRegs;r.conv_local=int(a.localSizeBytes);r.conv_shared=int(a.sharedSizeBytes);r.conv_binary=a.binaryVersion;
    r.commit_regs=b.numRegs;r.commit_local=int(b.localSizeBytes);r.commit_shared=int(b.sharedSizeBytes);r.commit_binary=b.binaryVersion;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.conv_blocks,time32,Threads,0));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.commit_blocks,commit_history,Threads,0));
    r.eligible=r.conv_binary==75&&r.commit_binary==75&&r.conv_local==0&&r.commit_local==0&&
        r.conv_shared==0&&r.commit_shared==0&&r.conv_blocks>0&&r.commit_blocks>0;
    return r;
}
bool native_gdn_conv_time32(float* history,const float* input,const float* weights,
                           float* output,int channels,int tokens,void* stream){
    if(channels!=Channels||tokens<3||tokens>8192||!stream)return false;
    const size_t row=size_t(Channels)*sizeof(float),data=size_t(tokens)*row,hist=3*row,weight=4*row;
    if(!span(history,hist)||!span(input,data)||!span(weights,weight)||!span(output,data)||
       overlap(history,hist,output,data)||overlap(history,hist,input,data)||overlap(history,hist,weights,weight)||
       overlap(output,data,input,data)||overlap(output,data,weights,weight))return false;
    if(!native_gdn_conv_time32_resources().eligible)return false;
    const auto s=static_cast<cudaStream_t>(stream);
    time32<<<dim3(Channels/Threads,(tokens+Time-1)/Time),Threads,0,s>>>(history,input,weights,output,tokens);
    check(cudaGetLastError());
    // A last time-block cannot write history itself: it may run before block0
    // has read the old history. Same-stream kernel order is the global boundary.
    commit_history<<<Channels/Threads,Threads,0,s>>>(history,input,tokens);check(cudaGetLastError());return true;
}
}
