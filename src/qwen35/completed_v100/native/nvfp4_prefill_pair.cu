#include "p2_vector_encode.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_gemm.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "nvfp4_decode8.hpp"
#include "nvfp4_prefill_pair.hpp"
#include "q8_power2_prefill.cuh"
#include <string>
namespace strata::completed_v100::qwen35::native {
void Nvfp4Projection::run_prefill_pair(const void* g,const float* sg,const void* u,const float* su,const float* x,
    float* packed,size_t capacity,float* activation,int rows,int columns,int tokens,cudaStream_t stream) {
    if(!q8_pp_pair_||!q8_pp_p2_||!q8_pp_decode8_||!g||!sg||!u||!su||!x||!packed||!activation||
       rows<1||columns<64||columns%64||!q8_pp_shape(tokens)||tokens>maximum_tokens_||columns>input_width_||
       size_t(rows)*columns>weight_values_||capacity<2*size_t(rows)*tokens||packed==x||packed==activation||x==activation)
        throw std::invalid_argument("NVFP4PPpair weight/input/output arena lease");
    nvfp4_decode_raw_half8(g,weights_,rows,columns,stream);
    auto* second=static_cast<unsigned char*>(weights_)+size_t(rows)*columns*2;
    nvfp4_decode_raw_half8(u,second,rows,columns,stream);
    if(q8_pp_vector_)q8p2_encode_vector(x,high_,size_t(tokens)*columns,stream);
    else q8p2_encode(x,high_,size_t(tokens)*columns,stream);
    const auto check=[](cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("NVFP4PPpaircuBLAS "+std::to_string(int(status)));};
    check(cublasSetStream(handle_,stream));check(cublasSetWorkspace(handle_,vendor_workspace_,VendorWorkspaceBytes));
    check(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_DEVICE));
    check(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,2*rows,tokens,columns,constants_,weights_,CUDA_R_16F,columns,
        high_,CUDA_R_16F,columns,constants_+3,packed,CUDA_R_32F,2*rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    nvfp4_pair_swiglu(packed,sg,su,activation,rows,tokens,stream);
}
bool Nvfp4Gemm::prefill_pair(Matrix gate,Matrix up,const float* x,float* packed,size_t capacity,float* act,int tokens,cudaStream_t s) {
    if(!q8_pp_pair()||!q8_pp_p2()||!q8_pp_decode8()||!q8_pp_shape(tokens)||
       gate.quant!=Quant::NVFP4||up.quant!=Quant::NVFP4)return false;
    original_matrix_bytes(gate);original_matrix_bytes(up);
    if(gate.layout!=BlockLayout::Nvfp4Scale||up.layout!=BlockLayout::Nvfp4Scale||gate.rows!=up.rows||gate.columns!=up.columns)
        throw std::invalid_argument("NVFP4PPpair matrix identity");
    projection_.run_prefill_pair(static_cast<const unsigned char*>(gate.data)+Nvfp4HeaderBytes,static_cast<const float*>(gate.data),
        static_cast<const unsigned char*>(up.data)+Nvfp4HeaderBytes,static_cast<const float*>(up.data),
        x,packed,capacity,act,gate.rows,gate.columns,tokens,s);calls_+=2;return true;
}
bool try_q8_prefill_pair(DenseGemm* service,Matrix gate,Matrix up,const float* x,float* packed,size_t capacity,float* act,int tokens,cudaStream_t s) {
    auto* provider=dynamic_cast<Nvfp4Gemm*>(service);
    return provider&&provider->prefill_pair(gate,up,x,packed,capacity,act,tokens,s);
}
}
