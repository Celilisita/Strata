// Fixed TG mapping of flash_attn_tile.cu's native F16 arithmetic: four warps,
// two heads per warp, one shared K/V load feeding both heads.
#include "attention_pairheads4.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdlib>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
constexpr int Heads=16, Group=8, Dim=256, Rows=64, Partial=258, Threads=128;
constexpr size_t SharedBytes=14336;
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__device__ float warp_maximum(float v) {
    for(int d=16;d;d>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,d));
    return v;
}
__device__ float warp_total(float v) {
    for(int d=16;d;d>>=1) v+=__shfl_xor_sync(0xffffffffu,v,d);
    return v;
}
__launch_bounds__(Threads,2)
__global__ void pair_tile(const float* q,const half2* k,const half2* v,
    const int32_t* step,int capacity,int width,int splits,float* partial,int32_t* status) {
    const int lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane;
    const int kv=blockIdx.x,split=blockIdx.y,base=split*Rows;
    const bool valid=step[kernels::kStepWidth]==width && width<=capacity &&
        step[kernels::kStepNKv]==width && step[kernels::kStepPos]==width-1 &&
        step[kernels::kStepNBid]==width/4;
    if(kv==0 && split==0 && tid==0)
        *status=valid ? kernels::kNativeFlashAttnSuccess : kernels::kNativeFlashAttnUnsupportedStep;
    if(!valid) return;
    __shared__ half2 qt[Group*128];
    __shared__ half2 cache[Rows*36];
    __shared__ half probability[Group*Rows];
    for(int i=tid;i<Group*128;i+=Threads) {
        const int d=i%128,head=kv*Group+i/128;
        const half2 value=__floats2half2_rn(q[head*Dim+2*d]*0.0625f,q[head*Dim+2*d+1]*0.0625f);
        qt[i]=__hmul2(value,__float2half2_rn(0.25f));
    }
    __syncthreads();
    float score[2][2]={};
    for(int chunk=0;chunk<128;chunk+=32) {
        for(int i=tid;i<Rows*32;i+=Threads) {
            const int row=i/32,d=i%32;
            cache[row*36+d]=base+row<width ? k[(base+row)*256+kv*128+chunk+d] : __float2half2_rn(0);
        }
        __syncthreads();
#pragma unroll
        for(int row=0;row<2;++row) {
#pragma unroll
            for(int d=0;d<32;++d) {
                const half2 value=cache[(lane+row*32)*36+d];
#pragma unroll
                for(int h=0;h<2;++h) {
                    const float2 p=__half22float2(__hmul2(value,qt[(2*warp+h)*128+chunk+d]));
                    score[h][row]+=p.x+p.y;
                }
            }
        }
        __syncthreads();
    }
    float maximum[2],denominator[2]={};
#pragma unroll
    for(int h=0;h<2;++h) {
        maximum[h]=-FLT_MAX/2;
#pragma unroll
        for(int row=0;row<2;++row) {
            const int cell=base+lane+row*32;
            score[h][row]=cell<width ? score[h][row]*4.0f+0.0f : __int_as_float(0xff800000u);
            maximum[h]=fmaxf(maximum[h],score[h][row]+3.0f*0.6931f);
        }
        maximum[h]=warp_maximum(maximum[h]);
#pragma unroll
        for(int row=0;row<2;++row) {
            const float weight=expf(score[h][row]-maximum[h]);
            denominator[h]+=weight;
            probability[(2*warp+h)*Rows+lane+row*32]=__float2half_rn(weight);
        }
        denominator[h]=warp_total(denominator[h]);
    }
    half2 accumulator[2][4]={};
    __syncthreads();
    for(int row0=0;row0<Rows;row0+=16) {
        for(int i=tid;i<16*128;i+=Threads) {
            const int cell=base+row0+i/128;
            cache[i]=cell<width ? v[cell*256+kv*128+i%128] : __float2half2_rn(0);
        }
        __syncthreads();
#pragma unroll
        for(int row=0;row<16;++row) {
            const half2 weights[2]={__half2half2(probability[2*warp*Rows+row0+row]),
                __half2half2(probability[(2*warp+1)*Rows+row0+row])};
#pragma unroll
            for(int d=0;d<4;++d) {
                const int pair=lane*2+(d&1)+(d/2)*64;
                const half2 value=cache[row*128+pair];
#pragma unroll
                for(int h=0;h<2;++h)
                    accumulator[h][d]=__hadd2(accumulator[h][d],__hmul2(value,weights[h]));
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int h=0;h<2;++h) {
        const int head=kv*Group+2*warp+h;
        float* out=partial+(head*splits+split)*Partial;
#pragma unroll
        for(int d=0;d<4;++d) {
            const int pair=lane*2+(d&1)+(d/2)*64;
            const float2 value=__half22float2(accumulator[h][d]);
            out[2*pair]=value.x;out[2*pair+1]=value.y;
        }
        if(lane==0) {out[Dim]=maximum[h];out[Dim+1]=denominator[h];}
    }
}
AttentionPairHeads4Resources query_resources() {
    AttentionPairHeads4Resources r{};r.threads=Threads;
    check(cudaGetDevice(&r.device));cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,r.device));
    r.major=p.major;r.minor=p.minor;r.multiprocessors=p.multiProcessorCount;r.shared_limit=p.sharedMemPerBlock;
    if(r.major!=7 || r.minor!=5 || r.multiprocessors!=72) {r.reason_mask=1;return r;}
    cudaFuncAttributes a{};check(cudaFuncGetAttributes(&a,pair_tile));
    r.binary_version=a.binaryVersion;r.registers=a.numRegs;r.max_threads=a.maxThreadsPerBlock;
    r.static_shared=a.sharedSizeBytes;r.local_bytes=a.localSizeBytes;
    if(r.binary_version!=75) r.reason_mask|=2;
    if(r.local_bytes) r.reason_mask|=4;
    if(r.static_shared!=SharedBytes || r.static_shared>r.shared_limit) r.reason_mask|=8;
    if(r.registers<1 || r.registers>255 || r.registers*Threads>p.regsPerBlock || r.max_threads<Threads) r.reason_mask|=16;
    if(!r.reason_mask) {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,pair_tile,Threads,0));
        if(r.active_blocks<1) r.reason_mask|=32;
    }
    return r;
}
}
AttentionPairHeads4Resources attention_pairheads4_resources() {
    static thread_local const AttentionPairHeads4Resources cached=query_resources();
    return cached;
}
bool tg_attention_pairheads4_enabled() {
    static const bool enabled=[] {
        const char* v=std::getenv("STRATA_QWEN35_TG_ATTN_PAIRHEADS4");
        if(!v) return false;
        if((v[0]!='0' && v[0]!='1') || v[1]!='\0')
            throw std::invalid_argument("TG_ATTN_PAIRHEADS4 must be0 or1");
        return v[0]=='1';
    }();
    return enabled;
}
bool try_launch_attention_pairheads4(const float* q,const uint16_t* k,const uint16_t* v,
    const int32_t* step,int capacity,int width,int splits,const uint16_t* mask,
    float* partial,int32_t* status,void* stream) {
    if(!q || !k || !v || !step || !partial || !status || !stream ||
        width<8192 || width>capacity || capacity>40960 || mask ||
        splits!=(width+Rows-1)/Rows || attention_pairheads4_resources().reason_mask) return false;
    pair_tile<<<dim3(2,splits),dim3(32,4),0,static_cast<cudaStream_t>(stream)>>>(
        q,reinterpret_cast<const half2*>(k),reinterpret_cast<const half2*>(v),
        step,capacity,width,splits,partial,status);
    check(cudaGetLastError());return true;
}
}
