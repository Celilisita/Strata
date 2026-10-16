#pragma once
#include "strata/qwen35/native/small_prefill.hpp"

namespace strata::qwen35::native {
// Lossless byte permutation: all [tile64][K256][row64] 16-byte headers,
// then original row-major [row][K256] 128-byte nibble payloads.
// Same 144 bytes/K256, original coefficients and unscaled integer codes.
// Private typed view; never admitted by generic Matrix operators.
struct Q4TileCodes { const void *data=nullptr;int rows=0,columns=0;size_t bytes=0; };
inline size_t q4_tile_codes_bytes(Q4TileCodes w){
    if(!w.data||reinterpret_cast<uintptr_t>(w.data)%16||w.rows<=0||w.rows%64||w.columns<=0||w.columns%256||
       w.bytes!=size_t(w.rows)*size_t(w.columns/256)*144)
        throw std::invalid_argument("invalid lossless tiled Q4 view");
    return w.bytes;
}
Q4TileCodes prepare_q4_tile_codes(Matrix,void *,cudaStream_t);
void small_prefill_q4_tiled_pair(Q4TileCodes,Q4TileCodes,const void *,float *,float *,float *,cudaStream_t,bool,void *,int);
}
