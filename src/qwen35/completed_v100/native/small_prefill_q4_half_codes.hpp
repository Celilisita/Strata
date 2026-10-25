#pragma once
#include "strata/qwen35/completed_v100/native/small_prefill.hpp"

namespace strata::completed_v100::qwen35::native {
// [tile64][K256][chunk128][row64][half128], followed by
// [tile64][K256][group8][row64][float2 d*scale,dmin*minimum].
struct Q4HalfCodeMatrix { const void *data=nullptr;int rows=0,columns=0;size_t bytes=0; };
// Different arithmetic/layout contract: codes are exact scale*q integers;
// metadata x is minimum_scale, y in group0/1 is original d/dmin.
struct Q4ScaledHalfCodeMatrix { Q4HalfCodeMatrix storage; };
constexpr size_t Q4HalfCodeBlockBytes=576;
inline size_t q4_half_code_matrix_bytes(Q4HalfCodeMatrix w){
    if(!w.data||w.rows<=0||w.rows%64||w.columns<=0||w.columns%256||
       w.bytes!=size_t(w.rows)*size_t(w.columns/256)*Q4HalfCodeBlockBytes)
        throw std::invalid_argument("invalid prepared Q4 half-code view or extent");
    return w.bytes;
}
size_t prepared_q4_half_codes_bytes(Matrix);
Q4HalfCodeMatrix prepare_q4_half_codes(Matrix,void *,cudaStream_t);
Q4ScaledHalfCodeMatrix prepare_q4_scaled_half_codes(Matrix,void *,cudaStream_t);
void small_prefill_q4_half_linear(Q4HalfCodeMatrix,const void *,float *,int,cudaStream_t,void *,size_t);
void small_prefill_q4_scaled_linear(Q4ScaledHalfCodeMatrix,const void *,float *,int,cudaStream_t,void *,size_t);
}
