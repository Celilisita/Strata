#include "p2_vector_encode.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_gemm.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "nvfp4_decode8.hpp"
#include "q8_power2_prefill.cuh"
namespace strata::completed_v100::qwen35::native {
void Nvfp4Projection::run_prefill_lt(Nvfp4PrefillLt& plan,const void* raw,const float* scale,const float* input,
    float* output,int rows,int columns,int tokens,cudaStream_t stream){
    if(!plan.supports(rows,columns,tokens)||!q8_pp_p2_||!q8_pp_decode8_||!raw||!scale||!input||!output||input==output||
       tokens>maximum_tokens_||columns>input_width_||size_t(rows)*columns>weight_values_)
        throw std::invalid_argument("NVFP4 Lt PP shape/policy/arena");
    nvfp4_decode_raw_half8(raw,weights_,rows,columns,stream);
    if(q8_pp_vector_)q8p2_encode_vector(input,high_,size_t(tokens)*columns,stream);
    else q8p2_encode(input,high_,size_t(tokens)*columns,stream);
    plan.project(weights_,high_,output,scale,constants_+3,vendor_workspace_,VendorWorkspaceBytes,stream);
}
}
