// Current Mini PP Down only. Preserve native Q8 codes/d and ordinary DS4 payload.
#if !defined(STRATA_DEEPSEEK_V4_GGML_API) || STRATA_DEEPSEEK_V4_GGML_API != 1
#error "Direct packed Down requires the actual production D4 private GGML ABI"
#endif
#include "native_down_direct_packed.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "mmq.cuh"
#include <cstdint>
namespace strata::prefill::mmq {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__launch_bounds__(256,1)
__global__ void direct_down_q8(const float* input,block_q8_1_mmq* output,
    int entries,const int32_t* ids) {
    const int lane=int(threadIdx.x)&31,warp=(int(blockIdx.x)*256+int(threadIdx.x))/32;
    if(warp>=entries*16)return; // Whole native32-value block/warp returns together.
    const int compact=warp%entries,qb=warp/entries,row=ids[compact];
    const float x=input[size_t(row)*512+qb*32+lane];float amax=fabsf(x);
#pragma unroll
    for(int d=16;d;d>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,d));
    const float scale=kernels::q8_1_finite(amax/127.0f);
    const int8_t code=kernels::q8_1_quant(x,scale,amax);int sum=int(code);
#pragma unroll
    for(int d=16;d;d>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,d);
    auto& dst=output[size_t(qb/4)*entries+compact];
    dst.qs[(qb%4)*32+lane]=code;
    if(lane==0){const half stored=__float2half_rn(scale);
        dst.ds4[qb%4]=make_half2(stored,__float2half_rn(__half2float(stored)*sum));}
}
}
DownDirectPackedResources native_down_direct_packed_resources() {
    DownDirectPackedResources r;int device=0;cudaDeviceProp p{};cudaFuncAttributes a{};
    check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&p,device));r.cc=p.major*10+p.minor;
    check(cudaFuncGetAttributes(&a,direct_down_q8));r.binary=a.binaryVersion;r.registers=a.numRegs;
    r.shared=a.sharedSizeBytes;r.local=a.localSizeBytes;r.max_threads=a.maxThreadsPerBlock;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,direct_down_q8,256,0));
    r.eligible=r.cc==75&&r.binary==75&&r.registers>0&&r.registers<=255&&r.shared==0&&r.local==0&&r.max_threads>=256&&r.active_blocks>0;
    return r;
}
void native_down_direct_packed(const float* input,void* packed,int entries,const int32_t* ids,void* stream) {
    if(!input||!packed||!ids||!stream||entries<8192||entries>16384||entries%8)
        throw std::invalid_argument("bounded direct Down producer");
    const auto st=static_cast<cudaStream_t>(stream);const int cells=entries*4;
    check(cudaMemsetAsync(static_cast<block_q8_1_mmq*>(packed)+cells,0,128*sizeof(block_q8_1_mmq),st));
    direct_down_q8<<<(entries*16+7)/8,256,0,st>>>(input,static_cast<block_q8_1_mmq*>(packed),entries,ids);
    check(cudaGetLastError()); // A queued CUDA error throws; never replay the old path.
}
} // namespace strata::prefill::mmq
