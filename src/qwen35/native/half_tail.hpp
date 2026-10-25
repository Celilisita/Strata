#pragma once
#include <fp4_gu_merge.hpp>
namespace strata::qwen35::native {
struct HalfTailResources {int registers,blocks,binary_version,kernel_count;size_t shared_bytes,local_bytes;};
void half_tail_configure();
HalfTailResources half_tail_resources();
void half_tail(const GuMergePlan&,PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,int,bool,float,cudaStream_t);
void half_tail_debug(const GuMergePlan&,PackedFp4,PackedFp4,Nvfp4Input,Nvfp4Input,float*,float*,int,bool,float,cudaStream_t);
}
