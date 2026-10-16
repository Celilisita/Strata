#pragma once
#include <cuda_runtime.h>

namespace strata::qwen35::native {
struct DecodeRounding;
struct DecodeCursor { int position,index,partitions;unsigned invalid; };
static_assert(sizeof(DecodeCursor)==16,"bounded device decode cursor extent");
struct DecodeCapture {
    DecodeCursor *cursor;
    int *ids;unsigned *flags;
    int capacity,partitions;
};
void initialize_decode_cursor(DecodeCursor *,int first,cudaStream_t);
void set_decode_position(int *,DecodeCursor *,int heads,DecodeRounding,cudaStream_t);
void record_decode_token(const int *,const unsigned *,DecodeCapture,cudaStream_t);
}
