#pragma once
#include <fp4_gu_fused.hpp>
namespace strata::qwen35::native {
void gutail_configure_kernel();
GuFusionResources gutail_resources();
void gutail_fused(PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,int,bool,float,cudaStream_t);
}
