#pragma once
#include <fp4_packed.hpp>
namespace strata::qwen35::native {
class DenseGemm;
struct GuFusionCounters {size_t input_bytes=0;uint64_t submissions=0;bool budget_fallback=false;};
GuFusionCounters nvfp4_gu_fusion_counters(DenseGemm &);
struct GuFusionResources {int registers,blocks;size_t shared_bytes,local_bytes;};
void fp4_gu_fused_configure();
GuFusionResources fp4_gu_fused_resources(bool half);
void fp4_gu_fused(PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,int,bool,float,cudaStream_t);
void fp4_gu_fused_debug(PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,float*,float*,int,bool,float,cudaStream_t);
void fp4_gu_codec_probe(const void*,const void*,Nvfp4Input,int,int,bool,float,cudaStream_t);
}
