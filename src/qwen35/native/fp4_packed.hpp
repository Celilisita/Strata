#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
namespace strata::qwen35::native {
// Typed physical representation; scales remain the canonical NVFP4 SF plane.
struct PackedFp4 {Nvfp4View view;};
struct PackedFp4Resources {int registers,blocks;size_t shared_bytes,local_bytes;};
PackedFp4 fp4_pack_values(Nvfp4View,void *,cudaStream_t);
void fp4_unpack_values(PackedFp4,void *,cudaStream_t);
PackedFp4Resources fp4_packed_resources(bool pair);
void fp4_packed_linear(PackedFp4,Nvfp4Input,float *,int,cudaStream_t);
void fp4_packed_pair(PackedFp4,PackedFp4,Nvfp4Input,float *,float *,float *,int,cudaStream_t);
}
