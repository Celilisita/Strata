#pragma once
#include "strata/qwen35/native/attention_gemm.hpp"
namespace strata::qwen35::native::pp_gqa {
struct Resources {int registers,blocks,local_bytes,shared_bytes;};
Resources resources();
size_t workspace(PrefillAttentionShape,int);
void run(DenseGemm&,const float*,const void*,const void*,float*,void*,size_t,
         PrefillAttentionShape,int,cudaStream_t);
}
