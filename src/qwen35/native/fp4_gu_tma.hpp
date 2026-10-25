#pragma once
#include "fp4_gu_fused.hpp"
#include <cuda.h>
namespace strata::qwen35::native {
class DenseGemm;
struct GuTmaCounters {uint64_t submissions=0;size_t plans=0;};
GuTmaCounters nvfp4_gu_tma_counters(DenseGemm &);
struct alignas(64) GuTmaMaps {CUtensorMap gate,up,input;};
struct GuTmaPlan {GuTmaMaps maps;const void *gate,*up,*input;int columns,rows,capacity;};
GuTmaPlan make_gu_tma_plan(PackedFp4,PackedFp4,Nvfp4Input);
void fp4_gu_tma_configure();
GuFusionResources fp4_gu_tma_resources(bool);
void fp4_gu_tma(const GuTmaPlan&,PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,int,bool,float,cudaStream_t);
void fp4_gu_tma_debug(const GuTmaPlan&,PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,float*,float*,int,bool,float,cudaStream_t);
}
