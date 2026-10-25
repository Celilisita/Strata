#pragma once
#include "strata/gemma4/native/vector_single.hpp"
namespace strata::gemma4::native {
// Exact210 GGUF payload bytes followed by14 zero padding bytes. The layout
// change permits aligned32-bit code reads; no scale/code value is requantized.
struct AlignedQ6View {
    const void* data=nullptr;size_t bytes=0;int rows=0,columns=0;
};
size_t aligned_q6_bytes(primitive::Matrix);
// Cold GPU-only repack, once per immutable weight tensor. Source and destination
// are separate; caller owns both and must keep them alive until stream completion.
AlignedQ6View pack_aligned_q6(primitive::Matrix,void*,size_t,cudaStream_t);
// Input remains the admitted40B single-Q8 activation format. No preparation,
// allocation, synchronization, raw-weight transfer or graph query occurs here.
void linear_aligned_q6(AlignedQ6View,SingleQ8View,float*,cudaStream_t);
}
