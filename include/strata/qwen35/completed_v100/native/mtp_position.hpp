#pragma once
#include "strata/qwen35/completed_v100/native/attention_decode.hpp"
namespace strata::completed_v100::qwen35::native {
struct MtpBatchCapture { const int *start;DecodeCursor *cursor;int maximum_partitions; };
void mtp_positions(int *out,const int *start,int offset,int tokens,int cache_shift,
                   DecodeCursor *,int heads,DecodeRounding,cudaStream_t);
void q8_cache_write_positions(const float *,void *,int dim,int heads,int tokens,
                             const int *,int shift,int capacity,cudaStream_t);
}
