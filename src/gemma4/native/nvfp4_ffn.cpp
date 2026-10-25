#include "strata/gemma4/native/nvfp4_ffn.hpp"
#include "strata/qwen35/native/ops.hpp"
#include "original_input_reuse.hpp"
#include "sm120_native_group_codec.hpp"
#include <cmath>
#include <stdexcept>

namespace strata::gemma4::native {
// Existing adopted Gemma ops.cu definitions. No SwiGLU provider is called.
void gelu_product(const float*,const float*,float*,std::size_t,cudaStream_t);
void residual_scale(const float*,const float*,const float*,float*,std::size_t,cudaStream_t);
namespace p=strata::qwen35::native;
Nvfp4Ffn::Nvfp4Ffn(Nvfp4ProjectionService& projections,int batch,float epsilon):
    projections_(projections),maximum_(batch),epsilon_(epsilon){
    if(batch<1||batch>8192||!(epsilon>0)||!std::isfinite(epsilon))
        throw std::invalid_argument("Gemma NVFP4 FFN precision/extent");
    auto floats=[&](int columns){return static_cast<float*>(storage_.allocate(std::size_t(batch)*columns*4));};
    normalized_=floats(5376);gate_=floats(21504);up_=floats(21504);
    activation_=floats(21504);down_=floats(5376);post_=floats(5376);
}
Nvfp4Ffn::~Nvfp4Ffn(){cudaStreamSynchronize(projections_.ordered_stream());}
void Nvfp4Ffn::run(const Nvfp4FfnWeights& weights,const float* input,float* output,int tokens,cudaStream_t stream){
    const auto body=[](p::Matrix matrix,int rows,int columns){return matrix.data&&matrix.rows==rows&&matrix.columns==columns&&
        ((matrix.quant==p::Quant::NVFP4&&(matrix.layout==p::BlockLayout::Nvfp4Planes||matrix.layout==p::BlockLayout::Nvfp4WarpFragments))||
         ((matrix.quant==p::Quant::Q4K||matrix.quant==p::Quant::Q6K)&&matrix.layout==p::BlockLayout::Gguf));};
    if(!input||!output||input==output||!stream||stream!=projections_.ordered_stream()||tokens<1||tokens>maximum_||
       !body(weights.gate,21504,5376)||!body(weights.up,21504,5376)||!body(weights.down,5376,21504)||
       !weights.pre_norm||!weights.post_norm||!weights.layer_scale)
        throw std::invalid_argument("Gemma NVFP4 FFN shape/math inventory");
    p::rms_norm(input,weights.pre_norm,normalized_,5376,tokens,epsilon_,stream);
    {
        OriginalInputReuseScope group(&projections_,normalized_,tokens,5376,stream,
            original_input_reuse_requested()&&tokens>3&&projections_.format()==GemmaWeightFormat::OriginalQuant);
        if(projections_.format()==GemmaWeightFormat::Nvfp4&&projections_.tiny_compensated()&&native_group_codec_tokens(tokens))
            projections_.project_group(normalized_,{{weights.gate,gate_},{weights.up,up_}},tokens,stream);
        else{
            projections_.project(weights.gate,normalized_,gate_,tokens,stream);
            projections_.project(weights.up,normalized_,up_,tokens,stream);
        }
    }
    gelu_product(gate_,up_,activation_,std::size_t(tokens)*21504,stream);
    projections_.project(weights.down,activation_,down_,tokens,stream);
    p::rms_norm(down_,weights.post_norm,post_,5376,tokens,epsilon_,stream);
    residual_scale(post_,input,weights.layer_scale,output,std::size_t(tokens)*5376,stream);
}
}
