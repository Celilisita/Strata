#pragma once
#include <fp4_compensated.hpp>
namespace strata::qwen35::native {
class DenseGemm;
struct ResidualNormCounters {uint64_t fused=0,fallback=0;};
ResidualNormCounters nvfp4_residual_norm_counters(DenseGemm&);
PackedFp4Resources fp4_residual_norm_resources();
void fp4_residual_norm(const float*,const float*,const float*,float*,float*,Nvfp4Input,int,float,float,cudaStream_t);
}
