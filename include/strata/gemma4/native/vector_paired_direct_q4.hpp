#pragma once
#include "strata/gemma4/native/vector_paired_word_q4.hpp"
namespace strata::gemma4::native {
struct PairedQ4DirectResources {int registers,static_shared_bytes,local_bytes,max_threads;};
PairedQ4DirectResources paired_q4_direct_resources();
// Independent candidate: nativeQ4 bases16B/Single40base8B, no shared/barrier/newshuffle.
void pair_single_q4_direct_t1(PairedQ4WordLease,float* gate,float* up,float* gelu,PairedQ4WordBinding live);
}
