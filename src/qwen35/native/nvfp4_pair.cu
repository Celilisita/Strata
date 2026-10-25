#include "strata/qwen35/native/nvfp4_gemm.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_q8_activation.cuh"
#include "nvfp4_q8_fast.cuh"
#include "nvfp4_q8_fastsoa.cuh"
#include "nvfp4_q8_halfscale.cuh"
namespace strata::qwen35::native {
void OriginalNvfp4Projection::run_pair(const void* gate,const float* sg,const void* up,const float* su,const float* input,
    float* g,float* u,float* activation,int rows,int columns,int tokens,cudaStream_t stream) {
    if(!q8_pair_||!q8_activation_||!gate||!sg||!up||!su||!input||!g||!u||!activation||
       rows<1||columns<64||columns%64||columns>input_width_||tokens<1||tokens>3||tokens>maximum_tokens_||
       input==g||input==u||input==activation||g==u||g==activation||u==activation)
        throw std::invalid_argument("NVFP4 Q8activation paired projection owner/extent");
    if(q8_pair_half_&&tokens==3&&rows==17408&&columns==5120) {
        nvfp4_q8hs_prepare(input,packed_input_,columns,tokens,stream);
        nvfp4_q8hs_project(gate,sg,up,su,packed_input_,g,u,activation,rows,columns,tokens,stream);
    }else if(q8_fastsoa_&&tokens==3) {
        nvfp4_q8fs_prepare(input,packed_input_,columns,tokens,stream);
        nvfp4_q8fs_project(gate,sg,up,su,packed_input_,g,u,activation,rows,columns,tokens,stream);
    }else if(q8_fast_&&tokens<=2) {
        nvfp4_q8f_prepare(input,packed_input_,columns,tokens,stream);
        nvfp4_q8f_project(gate,sg,up,su,packed_input_,g,u,activation,rows,columns,tokens,stream);
    }else{
        nvfp4_q8a_prepare(input,packed_input_,columns,tokens,stream);
        nvfp4_q8a_project(gate,sg,up,su,packed_input_,g,u,activation,rows,columns,tokens,stream);
    }
}
bool OriginalNvfp4Gemm::pair_original(Matrix gate,Matrix up,const float* input,float* g,float* u,float* act,int tokens,cudaStream_t stream) {
    if(!q8_pair()||!q8_activation()||tokens>3||gate.quant!=Quant::NVFP4||up.quant!=Quant::NVFP4)return false;
    original_matrix_bytes(gate);original_matrix_bytes(up);
    if(gate.layout!=OriginalNvfp4Scale||up.layout!=OriginalNvfp4Scale||gate.rows!=up.rows||gate.columns!=up.columns)
        throw std::invalid_argument("NVFP4 paired original matrices must share quant/layout/shape");
    projection_.run_pair(static_cast<const unsigned char*>(gate.data)+OriginalProviderHeaderBytes,static_cast<const float*>(gate.data),
        static_cast<const unsigned char*>(up.data)+OriginalProviderHeaderBytes,static_cast<const float*>(up.data),
        input,g,u,act,gate.rows,gate.columns,tokens,stream);calls_+=2;return true;
}
bool try_q8activation_pair(DenseGemm* service,Matrix gate,Matrix up,const float* input,float* g,float* u,float* act,int tokens,cudaStream_t stream) {
    auto* provider=dynamic_cast<OriginalNvfp4Gemm*>(service);
    if(!provider)throw std::invalid_argument("Original paired projection has no owned GPU provider");
    return provider->pair_original(gate,up,input,g,u,act,tokens,stream);
}
}
