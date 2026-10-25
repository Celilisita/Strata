// Native Q8 token-once repack/scatter. Ordinary MMQ still consumes its original compact layout.
#if !defined(STRATA_DEEPSEEK_V4_GGML_API) || STRATA_DEEPSEEK_V4_GGML_API != 1
#error "GU warp repack requires the actual production D4 private GGML ABI"
#endif
#include "native_gu_warp_repack.hpp"
#include "mmq.cuh"
#include <array>
#include <limits>
namespace strata::prefill::mmq {
namespace {
thread_local uint64_t submissions=0;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__launch_bounds__(128,1)
__global__ void gu_repack_scatter(const block_q8_1* input,block_q8_1_mmq* output,
    int rows,const int32_t* inverse) {
    const int lane=int(threadIdx.x)&31;
    const int work=(int(blockIdx.x)*128+int(threadIdx.x))/32;
    if(work>=rows*16)return; // Whole warp handles one real token's128-code block.
    const int token=work/16,kb=work%16,p=lane/8,w=lane%8,entries=rows*8;
    const auto* source=input+size_t(token)*64+4*kb;
    const int word=reinterpret_cast<const int*>(source[p].qs)[w];
    int sum=__dp4a(word,0x01010101,0); // Four signed native codes; exact bounded integer sum.
#pragma unroll
    for(int d=4;d;d>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,d,8);
    half2 metadata=make_half2(__float2half_rn(0.f),__float2half_rn(0.f));
    if(w==0){const half d=__low2half(source[p].ds);
        metadata=make_half2(d,__float2half_rn(__half2float(d)*sum));}
#pragma unroll
    for(int slot=0;slot<8;++slot){
        const int compact=inverse[token*8+slot];
        auto& dst=output[size_t(kb)*entries+compact];
        reinterpret_cast<int*>(dst.qs)[lane]=word;
        if(w==0)dst.ds4[p]=metadata;
    }
}
}
GuWarpRepackResources native_gu_warp_repack_resources() {
    GuWarpRepackResources r;int device=0;cudaDeviceProp p{};cudaFuncAttributes a{};
    ck(cudaGetDevice(&device));ck(cudaGetDeviceProperties(&p,device));r.cc=p.major*10+p.minor;
    ck(cudaFuncGetAttributes(&a,gu_repack_scatter));r.binary=a.binaryVersion;r.registers=a.numRegs;
    r.shared=a.sharedSizeBytes;r.local=a.localSizeBytes;r.max_threads=a.maxThreadsPerBlock;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,gu_repack_scatter,128,0));
    r.eligible=r.cc==75&&r.binary==75&&r.registers>0&&r.registers<=128&&r.shared==0&&
        r.local==0&&r.max_threads>=128&&r.active_blocks>0;return r;
}
bool native_gu_warp_repack_supported() {
    // Native service owns one fixed visible device/context. No query on each layer/request.
    thread_local const bool supported=native_gu_warp_repack_resources().eligible;return supported;
}
uint64_t native_gu_warp_repack_submissions(){return submissions;}
void native_gu_warp_repack(const void* input,void* output,int rows,const int32_t* inverse,void* stream) {
    if(!input||!output||!inverse||!stream||rows<1024||rows>2048)
        throw std::invalid_argument("bounded native GU repack/scatter");
    struct Span{uintptr_t begin,end;};
    auto span=[](const void* p,size_t n){const auto a=reinterpret_cast<uintptr_t>(p);
        if(a%4||n>std::numeric_limits<uintptr_t>::max()-a)throw std::invalid_argument("GU repack span");
        return Span{a,a+n};};
    const int entries=rows*8,cells=entries*16;
    const std::array<Span,3> spans={span(input,size_t(rows)*64*sizeof(block_q8_1)),
        span(output,size_t(cells+128)*sizeof(block_q8_1_mmq)),span(inverse,size_t(entries)*4)};
    for(size_t i=0;i<spans.size();++i)for(size_t j=0;j<i;++j)
        if(spans[i].begin<spans[j].end&&spans[j].begin<spans[i].end)
            throw std::invalid_argument("GU repack input/packed/control alias");
    const auto st=static_cast<cudaStream_t>(stream);
    ck(cudaMemsetAsync(static_cast<block_q8_1_mmq*>(output)+cells,0,128*sizeof(block_q8_1_mmq),st));
    gu_repack_scatter<<<(rows*16+3)/4,128,0,st>>>(static_cast<const block_q8_1*>(input),
        static_cast<block_q8_1_mmq*>(output),rows,inverse);
    ck(cudaGetLastError());++submissions; // A queued error throws; never replay the old path.
}
} // namespace strata::prefill::mmq
