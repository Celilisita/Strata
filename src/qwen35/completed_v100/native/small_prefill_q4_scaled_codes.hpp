#pragma once
#include "strata/qwen35/completed_v100/native/small_prefill.hpp"

namespace strata::completed_v100::qwen35::native {
// Per K256: d/dmin4B, eight half2 scale/min pairs32B, zero padding12B,
// then original unsigned four-bit128B payload. No weight-code expansion.
struct Q4ScaledCodeMatrix { const void *data=nullptr;int rows=0,columns=0;size_t bytes=0; };
inline size_t q4_scaled_code_matrix_bytes(Q4ScaledCodeMatrix w){
    if(!w.data||w.rows<=0||w.rows%4||w.columns<=0||w.columns%256||w.bytes!=size_t(w.rows)*size_t(w.columns/256)*176)
        throw std::invalid_argument("invalid compact scaled Q4 view or extent");
    return w.bytes;
}
size_t prepared_q4_scaled_codes_bytes(Matrix);
Q4ScaledCodeMatrix prepare_q4_scaled_codes(Matrix,void *,cudaStream_t);
void small_prefill_q4_compact_scaled_linear(Q4ScaledCodeMatrix,const void *,float *,int,cudaStream_t,void *,size_t);
}
