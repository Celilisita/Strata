#include "strata/qwen35/completed_v100/native/nvfp4.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "nvfp4_codec.cuh"
#include <climits>

namespace strata::completed_v100::qwen35::native {
namespace {
template<bool Diagnostic>
__global__ void decode(const unsigned char* source,half* output,size_t count,float* checked) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const auto* block=source+(index/64)*36;
    const half result=__float2half_rn(nvfp4::value(block,unsigned(index%64)));
    output[index]=result;
    if constexpr(Diagnostic)checked[index]=__half2float(result);
}
}
void nvfp4_decode_raw_half(const void* input,void* output,int rows,int columns,
                           cudaStream_t stream,float* diagnostic) {
    if(!input||!output||input==output||rows<1||columns<64||columns%64||
       size_t(rows)*columns>size_t(INT_MAX))
        throw std::invalid_argument("NVFP4 raw decoder extent/owner mismatch");
    const size_t count=size_t(rows)*columns;
    if(diagnostic)decode<true><<<unsigned((count+255)/256),256,0,stream>>>(
        static_cast<const unsigned char*>(input),static_cast<half*>(output),count,diagnostic);
    else decode<false><<<unsigned((count+255)/256),256,0,stream>>>(
        static_cast<const unsigned char*>(input),static_cast<half*>(output),count,nullptr);
    check_cuda(cudaGetLastError(),"Strata original NVFP4 block decode");
}
}
