#pragma once
#include <cuda_runtime.h>
namespace strata::completed_v100::qwen35::native {
// Preserve the original result/validity for the bulk host response, while
// ensuring an invalid result cannot index the following embedding gather.
void mtp_draft_link(const int *picked,const unsigned *invalid,int vocabulary,
                    int *next,int *record,unsigned *flags,cudaStream_t);
}
