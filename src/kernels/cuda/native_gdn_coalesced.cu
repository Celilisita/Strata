// Qualified Coalesced8 body; state/reduction/arithmetic are unchanged.
#include "strata/kernels/native_gdn_coalesced.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace strata::kernels {
namespace {
constexpr int S=128, Heads=32, KeyHeads=16, Columns=8, Pitch=Columns+1;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for(int d=16;d>0;d>>=1) v+=__shfl_xor_sync(0xffffffff,v,d,32);
    return v;
}
__global__ void __launch_bounds__(256,2) coalesced(
    float* __restrict__ state,const float* __restrict__ q,const float* __restrict__ k,
    const float* __restrict__ v,const float* __restrict__ gate,const float* __restrict__ beta,
    float* __restrict__ output,float scale) {
    const int head=blockIdx.x, tid=threadIdx.x, lane=tid%32, column=tid/32;
    const int first=blockIdx.y*Columns, col=first+column, qhead=head%KeyHeads;
    __shared__ float tile[S*Pitch];
    // Each warp accesses four complete 32-byte state sectors, rather than
    // 32 separate sectors. Pitch9 makes the transposed warp access conflict-free.
    for(int cell=tid;cell<S*Columns;cell+=blockDim.x) {
        const int i=cell/Columns,c=cell%Columns;
        tile[i*Pitch+c]=state[(size_t(i)*Heads+head)*S+first+c];
    }
    __syncthreads();
    float shard[4],kr[4],qr[4];
#pragma unroll
    for(int r=0;r<4;++r) {
        const int i=r*32+lane;
        shard[r]=tile[i*Pitch+column]; kr[r]=k[qhead*S+i]; qr[r]=q[qhead*S+i];
    }
    const float g=expf(gate[head]); float kv=0;
#pragma unroll
    for(int r=0;r<4;++r) kv+=shard[r]*kr[r];
    const float delta=(v[head*S+col]-g*warp_sum(kv))*beta[head]; float attn=0;
#pragma unroll
    for(int r=0;r<4;++r) {
        shard[r]=g*shard[r]+kr[r]*delta;
        attn+=shard[r]*qr[r];
    }
    const float result=warp_sum(attn);
    if(lane==0) output[head*S+col]=result*scale;
#pragma unroll
    for(int r=0;r<4;++r) tile[(r*32+lane)*Pitch+column]=shard[r];
    __syncthreads();
    for(int cell=tid;cell<S*Columns;cell+=blockDim.x) {
        const int i=cell/Columns,c=cell%Columns;
        state[(size_t(i)*Heads+head)*S+first+c]=tile[i*Pitch+c];
    }
}
bool valid_span(const void* p,size_t bytes) {
    const auto at=reinterpret_cast<uintptr_t>(p);
    return p&&at%sizeof(float)==0&&bytes<=UINTPTR_MAX-at;
}
bool overlap(const void* a,size_t an,const void* b,size_t bn) {
    const auto ap=reinterpret_cast<uintptr_t>(a),bp=reinterpret_cast<uintptr_t>(b);
    return ap<bp+bn&&bp<ap+an;
}
}
NativeGdnCoalescedResources native_gdn_coalesced_resources() noexcept {
    NativeGdnCoalescedResources r;cudaDeviceProp p{};
    if(cudaGetDevice(&r.device)!=cudaSuccess||cudaGetDeviceProperties(&p,r.device)!=cudaSuccess)return r;
    r.major=p.major;r.minor=p.minor;r.sms=p.multiProcessorCount;
    if(r.major!=7||r.minor!=5)return r;
    cudaFuncAttributes a{};int blocks=0;
    if(cudaFuncGetAttributes(&a,coalesced)!=cudaSuccess||
       cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,coalesced,256,0)!=cudaSuccess)return r;
    r.regs=a.numRegs;r.local=int(a.localSizeBytes);r.shared=int(a.sharedSizeBytes);
    r.blocks=blocks;r.binary=a.binaryVersion;
    r.eligible=r.local==0&&r.binary==75&&r.shared==4608&&r.blocks>0;
    return r;
}
bool native_gdn_coalesced_step(float* state,const float* q,const float* k,const float* v,
    const float* gate,const float* beta,float* output,const GdnShapes& shape,void* stream) {
    if(!stream||shape.S!=S||shape.h_k!=KeyHeads||shape.h_v!=Heads)return false;
    constexpr size_t state_bytes=size_t(S)*S*Heads*sizeof(float),qk_bytes=S*KeyHeads*sizeof(float);
    constexpr size_t output_bytes=S*Heads*sizeof(float),head_bytes=Heads*sizeof(float);
    if(!valid_span(state,state_bytes)||!valid_span(output,output_bytes)||
       overlap(state,state_bytes,output,output_bytes))return false;
    const void* inputs[]={q,k,v,gate,beta};
    const size_t sizes[]={qk_bytes,qk_bytes,output_bytes,head_bytes,head_bytes};
    for(int i=0;i<5;++i)if(!valid_span(inputs[i],sizes[i])||overlap(state,state_bytes,inputs[i],sizes[i])||
                           overlap(output,output_bytes,inputs[i],sizes[i]))return false;
    if(!native_gdn_coalesced_resources().eligible)return false;
    coalesced<<<dim3(Heads,S/Columns),256,0,static_cast<cudaStream_t>(stream)>>>(
        state,q,k,v,gate,beta,output,1.0f/sqrtf(float(S)));
    ck(cudaGetLastError());return true;
}
}
