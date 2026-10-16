#pragma once
#include "strata/qwen35/native/small_prefill.hpp"

namespace strata::qwen35::native {
// Exact Q4Pairs byte permutation. Headers: [tile64][K256][row64][16B].
// Codes: [tile64][K256][group_pair4][row64][32B], original low/high nibbles.
// A private cold canary view, never accepted by generic Matrix operators.
struct Q4GroupCodes { const void *data=nullptr;int rows=0,columns=0;size_t bytes=0; };
inline size_t q4_group_codes_bytes(Q4GroupCodes w){
    if(!w.data||reinterpret_cast<uintptr_t>(w.data)%16||w.rows<=0||w.rows%64||w.columns<=0||w.columns%256||
       w.bytes!=size_t(w.rows)*size_t(w.columns/256)*144)
        throw std::invalid_argument("invalid lossless group Q4 view");
    return w.bytes;
}
Q4GroupCodes prepare_q4_group_codes(Matrix,void *,cudaStream_t);
// Diagnostic: exact comparison of every source uint4, using borrowed 8B scratch.
void verify_q4_group_codes(Matrix,Q4GroupCodes,void *,cudaStream_t);
void small_prefill_q4_group_pair(Q4GroupCodes,Q4GroupCodes,const void *,float *,float *,float *,cudaStream_t,bool,void *);
}
