#pragma once
#include <cublasLt.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
namespace strata::qwen35::native {
inline bool half_tactic_shape(int rows,int columns,int tokens){return rows==17408&&columns==5120&&tokens==8192;}
inline bool half_tactic_requested(){
    const char *flag=std::getenv("STRATA_NVFP4_HALF_TACTIC");
    if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("NVFP4 half tactic mode");
    return !flag||flag[0]!='0';
}
inline bool half_tactic_matches(const cublasLtMatmulAlgo_t &algorithm,bool alternate){
    // Public configuration is stable; opaque descriptor bytes contain private state.
    if(cublasLtGetVersion()!=130000)return false;
    using Attribute=decltype(CUBLASLT_ALGO_CONFIG_ID);
    const std::array<std::pair<Attribute,int>,7> expected{{
        {CUBLASLT_ALGO_CONFIG_ID,70},{CUBLASLT_ALGO_CONFIG_TILE_ID,20},
        {CUBLASLT_ALGO_CONFIG_SPLITK_NUM,alternate?1:-2},
        {CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,alternate?0:2},
        {CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,0},{CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,0},
        {CUBLASLT_ALGO_CONFIG_STAGES_ID,37}
    }};
    for(const auto &field:expected){
        int value=-1;size_t bytes=0;
        if(cublasLtMatmulAlgoConfigGetAttribute(&algorithm,field.first,&value,sizeof(value),&bytes)!=CUBLAS_STATUS_SUCCESS||
            bytes!=sizeof(value)||value!=field.second)return false;
    }
    return true;
}
class DenseGemm;
uint64_t nvfp4_half_tactic_submissions(DenseGemm &);
}
