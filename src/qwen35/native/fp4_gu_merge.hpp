#pragma once
#include "fp4_gu_tma.hpp"
#include <cstddef>
namespace strata::qwen35::native {
class DenseGemm;
struct GuMergeCounters {uint64_t submissions=0;size_t plans=0;};
GuMergeCounters nvfp4_gu_merge_counters(DenseGemm &);
// Distinct type prevents passing a64feature TensorMap to the128feature kernel.
struct GuMergePlan {GuTmaMaps maps;const void *gate,*up,*input;int columns,rows,capacity;};
static_assert(sizeof(GuMergePlan)==448&&offsetof(GuMergePlan,gate)==384&&
    offsetof(GuMergePlan,up)==392&&offsetof(GuMergePlan,input)==400&&offsetof(GuMergePlan,capacity)==416,
    "retained qualified CUDA descriptor ABI");
GuMergePlan make_gu_merge_plan(PackedFp4,PackedFp4,Nvfp4Input);
void fp4_gu_merge_configure();
GuFusionResources fp4_gu_merge_resources(bool);
void fp4_gu_merge(const GuMergePlan&,PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,int,bool,float,cudaStream_t);
void fp4_gu_merge_debug(const GuMergePlan&,PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,float*,float*,int,bool,float,cudaStream_t);
}
