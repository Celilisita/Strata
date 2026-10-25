#include "strata/qwen35/completed_v100/native/recurrent.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_gemm.hpp"
namespace strata::completed_v100::qwen35::native {
void RecurrentLayer::use_original_gemm(DenseGemm* service) {
    if(!dynamic_cast<Nvfp4Gemm*>(service)||!pp_||!tg_)
        throw std::invalid_argument("Original quantized GPU provider requires both phase owners");
    pp_gemm_=service;pp_->use_original_gemm(service);tg_->use_original_gemm(service);
}
void Ffn::use_original_gemm(DenseGemm* service) {
    if(!dynamic_cast<Nvfp4Gemm*>(service)||!original_projection(gate_)||
       !original_projection(up_)||!original_projection(down_)||tiled_pair_||group_pair_)
        throw std::invalid_argument("Original GPU FFN provider requires an unleased compatible inventory");
    pp_gemm_=service;
}

}
