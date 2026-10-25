#include "pp_prefetch.hpp"
#include "prefill_down_tensor.hpp"
#include "p2_vector_encode.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_gemm.hpp"
#include "nvfp4_decode8.hpp"
#include "q8_power2_prefill.cuh"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "math.cuh"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <string>
namespace strata::completed_v100::qwen35::native {
namespace n=strata::completed_v100::qwen35::native;
namespace {
template<bool Store>__global__ void epilogue(const float* x,const float* sg,const float* su,float* activation,
    half* encoded,int rows,size_t count){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const size_t at=(i/rows)*size_t(2*rows)+i%rows;
    const float g=__fmul_rn(x[at],*sg),u=__fmul_rn(x[at+rows],*su);
    const float value=n::swiglu_value(g,u);
    if constexpr(Store)activation[i]=value;
    float maximum=fabsf(value);
    for(int d=8;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,16));
    const unsigned invalid=__ballot_sync(0xffffffff,!isfinite(value)||fabsf(value)>65024.f);
    const bool bad=(invalid&(0xffffu<<(threadIdx.x&16)))!=0;
    int exponent=0;const float fraction=frexpf(maximum/127.f,&exponent);
    const int power=max(-14,min(9,exponent-(fraction==.5f)));
    const float scale=maximum>0.f?ldexpf(1.f,power):ldexpf(1.f,-14);
    const int code=max(-127,min(127,__float2int_rn(value/scale)));
    encoded[i]=__float2half_rn(bad?CUDART_NAN_F:float(code)*scale);
}

}
void swiglu_encode(const float* x,const float* sg,const float* su,float* activation,void* encoded,
    int rows,int tokens,cudaStream_t stream){
    if(!x||!sg||!su||!encoded||static_cast<const void*>(x)==encoded||x==activation||
       static_cast<void*>(activation)==encoded||rows<64||rows%64||tokens<1||tokens>8192)
        throw std::invalid_argument("PP107 packed/activation/encoded lease");
    const size_t count=size_t(rows)*tokens;
    if(activation)epilogue<true><<<unsigned((count+127)/128),128,0,stream>>>(x,sg,su,activation,static_cast<half*>(encoded),rows,count);
    else epilogue<false><<<unsigned((count+127)/128),128,0,stream>>>(x,sg,su,nullptr,static_cast<half*>(encoded),rows,count);
    n::check_cuda(cudaGetLastError(),"PP107 fused SwiGLU/P2 encode");
}
void Nvfp4Projection::run_prefill_fused(Nvfp4PrefillLt* plan,const void* g,const float* sg,const void* u,const float* su,
    const void* d,const float* sd,const float* x,float* packed,size_t capacity,float* output,int rows,int columns,int tokens,cudaStream_t stream){
    if(!q8_pp_pair_||!q8_pp_p2_||!q8_pp_decode8_||!g||!sg||!u||!su||!d||!sd||!x||!packed||!output||
       rows<64||rows%64||columns<64||columns%64||!q8_pp_shape(tokens)||tokens>maximum_tokens_||
       (std::max)(rows,columns)>input_width_||size_t(rows)*columns>weight_values_||capacity<2*size_t(rows)*tokens||
       packed==x||packed==output||x==output||(plan&&!plan->supports(columns,rows,tokens)))
        throw std::invalid_argument("NVFP4 fused FFN shape/policy/arena lease");
    nvfp4_decode_raw_half8(g,weights_,rows,columns,stream);
    nvfp4_decode_raw_half8(u,static_cast<unsigned char*>(weights_)+size_t(rows)*columns*2,rows,columns,stream);
    if(q8_pp_vector_)q8p2_encode_vector(x,high_,size_t(tokens)*columns,stream);
    else q8p2_encode(x,high_,size_t(tokens)*columns,stream);
    const auto check=[](cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("NVFP4 fused FFN cuBLAS "+std::to_string(int(status)));};
    check(cublasSetStream(handle_,stream));check(cublasSetWorkspace(handle_,vendor_workspace_,VendorWorkspaceBytes));
    check(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_DEVICE));
    check(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,2*rows,tokens,columns,constants_,weights_,CUDA_R_16F,columns,
        high_,CUDA_R_16F,columns,constants_+3,packed,CUDA_R_32F,2*rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // The Gate/Up GEMM has consumed high_; the same stream now reuses it for Down.
    if(q8_pp_vector_)swiglu_encode_vector(packed,sg,su,high_,rows,tokens,stream);
    else swiglu_encode(packed,sg,su,nullptr,high_,rows,tokens,stream);
    nvfp4_decode_raw_half8(d,weights_,columns,rows,stream);
    if(q8_pp_long_&&tokens==8192)prefill_down_tensor(weights_,high_,output,columns,rows,tokens,sd,constants_+3,stream);
    else if(plan)plan->project(weights_,high_,output,sd,constants_+3,vendor_workspace_,VendorWorkspaceBytes,stream);
    else check(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,columns,tokens,rows,sd,weights_,CUDA_R_16F,rows,
        high_,CUDA_R_16F,rows,constants_+3,output,CUDA_R_32F,columns,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
bool Nvfp4Gemm::prefill_fused(Matrix gate,Matrix up,Matrix down,const float* input,float* packed,size_t capacity,float* output,int tokens,cudaStream_t stream){
    if(!q8_pp_fused_||!q8_pp_pair()||!q8_pp_p2()||!q8_pp_decode8()||!q8_pp_shape(tokens)||
       gate.quant!=Quant::NVFP4||up.quant!=Quant::NVFP4||down.quant!=Quant::NVFP4)return false;
    original_matrix_bytes(gate);original_matrix_bytes(up);original_matrix_bytes(down);
    if(gate.layout!=BlockLayout::Nvfp4Scale||up.layout!=gate.layout||down.layout!=gate.layout||gate.rows!=up.rows||
       gate.columns!=up.columns||down.rows!=gate.columns||down.columns!=gate.rows)throw std::invalid_argument("NVFP4 fused FFN matrix identity");
    const auto raw=[](Matrix m){return static_cast<const unsigned char*>(m.data)+Nvfp4HeaderBytes;};
    const bool library=q8_pp_lt_&&pp_lt_.supports(down.rows,down.columns,tokens);
    projection_.run_prefill_fused(library?&pp_lt_:nullptr,raw(gate),static_cast<const float*>(gate.data),raw(up),static_cast<const float*>(up.data),
        raw(down),static_cast<const float*>(down.data),input,packed,capacity,output,gate.rows,gate.columns,tokens,stream);
    calls_+=3;++pp_fused_calls_;if(q8_pp_tail()&&tokens==6144)++pp_tail_calls_;if(q8_pp_long()&&tokens==8192)++pp_long_calls_;if(library)++pp_lt_calls_;return true;
}
bool try_q8_prefill_fused(DenseGemm* service,Matrix gate,Matrix up,Matrix down,const float* x,float* packed,size_t capacity,float* output,int tokens,cudaStream_t stream){
    if(try_pp_prefetch_ffn(service,gate,up,down,x,packed,capacity,output,tokens,stream))return true;
    auto* provider=dynamic_cast<Nvfp4Gemm*>(service);
    return provider&&provider->prefill_fused(gate,up,down,x,packed,capacity,output,tokens,stream);
}
}
