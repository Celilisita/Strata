#include "nvfp4_decode8.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_codec.cuh"
#include "nvfp4_half_scale.cuh"
#include <climits>
namespace strata::qwen35::native {
namespace {
__device__ __forceinline__ unsigned bits(unsigned nibble) {
    const unsigned m=nibble&7;
    const unsigned magnitude=m==0?0:m==1?0x3c00u:0x3c00u+(m<<9);
    return magnitude|(m?(nibble&8)<<12:0);
}
__device__ __forceinline__ half2 pair(unsigned word,int shift,half2 scale) {
    const unsigned raw=bits((word>>shift)&15)|(bits((word>>(shift+8))&15)<<16);
    return __hmul2(__halves2half2(__ushort_as_half(static_cast<unsigned short>(raw)),__ushort_as_half(static_cast<unsigned short>(raw>>16))),scale);
}
__device__ __forceinline__ unsigned raw(half2 v) {
    return unsigned(__half_as_ushort(__low2half(v)))|(unsigned(__half_as_ushort(__high2half(v)))<<16);
}
__global__ void decode8(const unsigned char* input,half2* out,size_t groups) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=groups*2)return;
    const size_t group=index/2;const unsigned shift=(index&1)*4;
    const auto* block=input+(group/4)*36;const unsigned local=group&3;
    const auto* codes=reinterpret_cast<const unsigned*>(block+4+local*8);
    const unsigned low=__ldg(codes),high=__ldg(codes+1);
    const half d=__float2half_rn(nvfp4::half_scale(__ldg(block+local)));
    const half2 scale=__halves2half2(d,d);
    // Each lane writes eight consecutive FP16 values in one aligned128-bit
    // transaction; adjacent lanes cover adjacent half-groups without gaps.
    reinterpret_cast<uint4*>(out)[index]=make_uint4(raw(pair(low,shift,scale)),raw(pair(low,shift+16,scale)),
        raw(pair(high,shift,scale)),raw(pair(high,shift+16,scale)));
}
}
void nvfp4_decode_raw_half8(const void* x,void* y,int rows,int columns,cudaStream_t s) {
    if(!x||!y||x==y||reinterpret_cast<uintptr_t>(x)%4||reinterpret_cast<uintptr_t>(y)%16||rows<1||columns<64||columns%64||size_t(rows)*columns>size_t(INT_MAX))throw std::invalid_argument("NVFP4contiguous8decode span/alignment");
    const size_t groups=size_t(rows)*columns/16;
    decode8<<<unsigned((groups*2+255)/256),256,0,s>>>(static_cast<const unsigned char*>(x),static_cast<half2*>(y),groups);
    check_cuda(cudaGetLastError(),"PP082vector8FP16decode");
}
}
