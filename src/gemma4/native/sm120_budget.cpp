#include "strata/gemma4/native/sm120_budget.hpp"
#include "pp_tc_attention.hpp"
#include "strata/qwen35/native/weights.hpp"
#include "strata/qwen35/spec.hpp"
#include "sm120_tiny_projection.hpp"
#include "original_quant_projection.hpp"
#include "sm120_rope_table.hpp"
#include "sm120_global_flash.hpp"
#include "whole_history.hpp"
#include "sm120_cached_global.hpp"
#include "global_keys1024_binding.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
namespace strata::gemma4::native {
Sm120Budget estimate_sm120_budget(const GgufFile& file,const Spec& spec,int batch,int total){
    if(batch<1||batch>8192||total<0||total>spec.context)throw std::invalid_argument("Gemma budget extent");
    namespace p=strata::qwen35::native;Sm120Budget b;
    const auto plan=plan_nvfp4_body(file,spec);
    const bool original=plan.format==GemmaWeightFormat::OriginalQuant;
    const bool native_backend=original||plan.format==GemmaWeightFormat::Nvfp4;
    const bool keys1024=global_keys1024::pp_tc_requested();
    if(keys1024&&(!native_backend||!pp_tc_requested()||batch!=2048||global_flash_requested()||
       whole_history_requested()||cached_global_requested()))
        throw std::invalid_argument("Gemma global Keys1024 budget requires a native backend, PP_TC, batch2048 and other PP leaves off");
    if(cached_global_requested()&&(!original||!pp_tc_requested()||global_flash_requested()||whole_history_requested()))
        throw std::invalid_argument("Gemma cached global budget requires original_quant, PP_TC and other PP leaves off");
    if(whole_history_requested()&&(!original||global_flash_requested()))
        throw std::invalid_argument("Gemma wholehistory budget requires original_quant and global_flash=0");
    if(global_flash_requested()&&!original)
        throw std::invalid_argument("Gemma global flash budget requires original_quant");
    const bool tiny=gemma_tiny_compensated_requested();
    if(original&&tiny)throw std::invalid_argument("Gemma original budget forbids NVFP4 tiny policy");
    const auto tiny_owned=tiny?gemma_tiny_canonical_scratch(plan)+gemma_tiny_low_bytes(plan):0;
    const auto bytes=[&](const TensorInfo* t){if(!t)throw std::invalid_argument("Gemma budget parameter");return qwen35::tensor_bytes(file,*t);};
    const auto initial=bytes(spec.embedding)+bytes(spec.output_norm)+bytes(spec.rope_frequencies);
    b.weights=initial;
    for(const auto& layer:spec.layers)for(auto* t:{layer.attn_norm,layer.attn_post,layer.q_norm,layer.k_norm,layer.ffn_norm,layer.ffn_post,layer.scale})b.weights+=bytes(t);
    const int rows=std::max(16,batch),columns=plan.maximum_columns;
    b.encoded_input=original?original_projection_input_bytes(batch,columns):
        p::nvfp4_value_bytes(rows,columns)+p::nvfp4_scale_bytes(rows,columns);
    std::size_t loaded=0;
    for(const auto& selected:plan.tensors){
        const auto& t=*selected.tensor;const int r=int(t.shape[1]),c=int(t.shape[0]);
        const auto resident=original?bytes(&t):p::nvfp4_value_bytes(r,c)+p::nvfp4_scale_bytes(r,c);
        loaded+=resident;
        const auto transient=original?0:std::max(bytes(&t)+sizeof(unsigned),tiny?resident:0);
        b.constructor_peak=std::max(b.constructor_peak,initial+b.encoded_input+tiny_owned+loaded+transient);
    }
    b.weights+=loaded;
    b.provider=original?0:
        32u*1024*1024+std::size_t(plan.maximum_rows)*16*4+tiny_owned;
    b.ffn=std::size_t(batch)*3*(5376+21504)*4;
    b.model_io=std::size_t(batch)*(6*5376+2*32*512+2*16*256)*4+std::size_t(batch)*4+5376*4+262144*4+8;
    if(pp_rope_table_requested()&&batch>3)b.model_io+=rope_table_bytes(batch);
    b.vendor_pool_bound=original?original_projection_pool_bound_bytes():0;
    b.static_owned=b.weights+b.encoded_input+b.provider+b.ffn+b.model_io;
    b.constructor_peak=std::max(b.constructor_peak,b.static_owned+b.vendor_pool_bound);
    if(total){
        b.kv=std::size_t(435200)*std::min(total,1024+batch)+std::size_t(43520)*total;
        const auto split=std::size_t((total+127)/128)*32*std::min(3,batch)*514*4;
        auto arena=std::max(std::size_t(92340224),split);
        if(pp_tc_requested())arena=std::max(arena,pp_tc_workspace_bytes(batch));
        if(keys1024)arena=std::max(arena,global_keys1024::pp_tc_workspace_bytes(batch));
        if(global_flash_requested())arena=std::max(arena,global_flash_workspace_bytes(batch,total));
        if(whole_history_requested())arena=std::max(arena,whole_history_workspace_bytes(batch,total));
        if(cached_global_requested())arena=std::max(arena,cached_global_workspace_bytes(batch,total));
        b.attention=arena+32u*1024*1024;
        b.request=b.kv+b.attention+b.vendor_pool_bound;
    }
    return b;
}
void admit_sm120_budget(std::size_t additional,const char* phase){
    std::size_t free=0,total=0;strata::qwen35::native::check_cuda(cudaMemGetInfo(&free,&total),"Gemma actual free GPU budget");
    const auto reserve=strata::qwen35::native::DeviceStorage::reserve_bytes();
    if(additional>free||free-additional<reserve)
        throw std::runtime_error(std::string("Gemma GPU budget ")+phase+": additional="+std::to_string(additional)+
            " free="+std::to_string(free)+" reserve="+std::to_string(reserve)+" total="+std::to_string(total));
}
}
