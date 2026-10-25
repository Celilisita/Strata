#include "strata/gemma4/native/sm120_api.hpp"
#include <exception>
namespace strata::gemma4::native {
bool identify_sm120_gemma(const GgufFile& file,Sm120FamilyInfo& info,std::string& error){
    try{const auto spec=bind_sm120_spec(file);plan_nvfp4_body(file,spec);
        info={};info.variant=gemma_weight_variant(gemma_weight_format_requested());error.clear();return true;
    }catch(const std::exception& failure){error=failure.what();return false;}
}
std::unique_ptr<Sm120Model> make_sm120_gemma(const GgufFile& file,int batch,cudaStream_t stream){
    return std::make_unique<Sm120Model>(file,batch,stream);
}
}
