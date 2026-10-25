#pragma once
#include "strata/gemma4/native/sm120_model.hpp"
#include <memory>
#include <string>
namespace strata::gemma4::native {
struct Sm120FamilyInfo {
    const char* architecture="gemma4";
    const char* family="gemma4_dense";
    const char* variant="gemma4_31b_nvfp4";
    int layers=60,hidden=5376,vocabulary=262144,logical_context=262144;
    bool mtp=false;
};
// Common-family registration can call this CPU-only contract before dispatch.
// It never falls through to a Qwen model executor.
bool identify_sm120_gemma(const GgufFile&,Sm120FamilyInfo&,std::string& error);
std::unique_ptr<Sm120Model> make_sm120_gemma(const GgufFile&,int batch,cudaStream_t);
}
