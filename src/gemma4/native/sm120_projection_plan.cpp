#include "strata/gemma4/native/sm120_projection_plan.hpp"
#include "strata/qwen35/spec.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

namespace strata::gemma4::native {
GemmaWeightFormat gemma_weight_format_requested(){
    const char* raw=std::getenv("STRATA_GEMMA_WEIGHT_FORMAT");const std::string value=raw?raw:"nvfp4";
    if(value=="nvfp4")return GemmaWeightFormat::Nvfp4;
    if(value=="original_quant")return GemmaWeightFormat::OriginalQuant;
    throw std::invalid_argument("STRATA_GEMMA_WEIGHT_FORMAT requires nvfp4/original_quant");
}
const char* gemma_weight_format_name(GemmaWeightFormat format){
    return format==GemmaWeightFormat::OriginalQuant?"original_quant":"nvfp4";
}
const char* gemma_weight_variant(GemmaWeightFormat format){
    return format==GemmaWeightFormat::OriginalQuant?"gemma4_31b_original_quant":"gemma4_31b_nvfp4";
}
Nvfp4BodyPlan plan_nvfp4_body(const GgufFile& file,const Spec& spec){
    if(spec.layers.size()!=60||spec.hidden!=5376||spec.intermediate!=21504||
       spec.heads!=32||spec.vocabulary!=262144||spec.window!=1024||
       !spec.embedding||spec.output!=spec.embedding||
       spec.embedding->type!=14)
        throw std::invalid_argument("Gemma SM120 requires admitted Dense31B and original tied embedding/head");
    Nvfp4BodyPlan plan;plan.format=gemma_weight_format_requested();
    const bool original=plan.format==GemmaWeightFormat::OriginalQuant;int q4=0,q6=0;
    std::unordered_set<const TensorInfo*> admitted;
    auto add=[&](const TensorInfo* tensor,ProjectionRole role,int layer){
        if(!tensor||file.find(tensor->name)!=tensor||(original?(tensor->type!=12&&tensor->type!=14):tensor->type!=40)||tensor->shape.size()!=2||
           !tensor->shape[0]||tensor->shape[0]%256||!tensor->shape[1]||
           tensor->shape[0]>21504||tensor->shape[1]>21504||
           tensor->name.size()<7||tensor->name.substr(tensor->name.size()-7)!=".weight"||
           !admitted.insert(tensor).second)
            throw std::invalid_argument("Gemma complete explicitly selected body inventory");
        const auto bytes=qwen35::tensor_bytes(file,*tensor);
        if(original){
            const auto stride=tensor->type==12?144:210;
            if(bytes!=tensor->shape[1]*(tensor->shape[0]/256)*stride)
                throw std::invalid_argument("Gemma original Q4_K/Q6_K GGUF block extent");
            q4+=tensor->type==12;q6+=tensor->type==14;
        }else{
            const auto* scale=file.find(tensor->name.substr(0,tensor->name.size()-7)+".scale");
            if(!scale||scale->type!=0||scale->elements()!=1||qwen35::tensor_bytes(file,*scale)!=sizeof(float))
                throw std::invalid_argument("Gemma NVFP4 requires one FP32 global weight scale");
            float global;std::memcpy(&global,file.tensor_data(*scale),sizeof(global));
            if(!(global>0)||!std::isfinite(global))throw std::invalid_argument("Gemma NVFP4 global scale");
            if(bytes!=tensor->shape[1]*(tensor->shape[0]/64)*36)
                throw std::invalid_argument("Gemma NVFP4 GGUF block extent");
        }
        plan.maximum_rows=std::max(plan.maximum_rows,int(tensor->shape[1]));
        plan.maximum_columns=std::max(plan.maximum_columns,int(tensor->shape[0]));
        plan.gguf_bytes+=bytes;plan.tensors.push_back({tensor,role,layer});
    };
    for(int i=0;i<60;++i){
        const auto& layer=spec.layers[i];const bool local=i%6!=5;
        if(layer.sliding!=local||layer.shared_raw_kv==local||
           layer.head_dim!=(local?256:512)||layer.kv_heads!=(local?16:4)||
           (!local&&layer.v!=layer.k))throw std::invalid_argument("Gemma per-layer KV math identity");
        add(layer.q,ProjectionRole::Query,i);add(layer.k,ProjectionRole::Key,i);
        if(local)add(layer.v,ProjectionRole::Value,i);
        add(layer.out,ProjectionRole::AttentionOutput,i);add(layer.gate,ProjectionRole::Gate,i);
        add(layer.up,ProjectionRole::Up,i);add(layer.down,ProjectionRole::Down,i);
    }
    if(plan.tensors.size()!=410)throw std::logic_error("Gemma unique body count");
    if(original&&(q4!=355||q6!=55||file.tensors().size()!=833))
        throw std::invalid_argument("Gemma original requires355Q4_K/55Q6_K body and833tensors");
    for(const auto& tensor:file.tensors()){
        if(original&&tensor.type==40)throw std::invalid_argument("Gemma original forbids NVFP4 payload");
        if(!original&&tensor.type==40&&!admitted.count(&tensor))
            throw std::invalid_argument("Gemma unexpected NVFP4 payload outside selected body");
        if(original&&(tensor.type==12||tensor.type==14)&& &tensor!=spec.embedding&&!admitted.count(&tensor))
            throw std::invalid_argument("Gemma original quantized payload outside admitted inventory");
    }
    return plan;
}
Sm120Attention sm120_attention_choice(int tokens,int past,int reserved,int batch){
    if(tokens<1||tokens>batch||batch<1||batch>8192||past<0||reserved<1||reserved>262144||
       past>reserved||tokens>reserved-past)
        throw std::invalid_argument("Gemma SM120 causal request extent");
    return tokens<=3?Sm120Attention::PackedQ8Split:Sm120Attention::Float32Tiled;
}
}
