#pragma once
#include "strata/gemma4/native/vector_single.hpp"
#include <cstdint>
namespace strata::gemma4::native {
struct PairedQ4WordBinding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct PairedQ4WordLease {
    primitive::Matrix gate,up;
    const float* normalized=nullptr;
    SingleQ8View input;
    PairedQ4WordBinding binding;
};
// Research T1 GateUp only; exact5376→21504 nativeQ4/Single40, original4Kstripes.
// No new ownership, packing, allocation, synchronization or Model policy.
void pair_single_q4_word_t1(PairedQ4WordLease,float* gate,float* up,float* gelu,
    PairedQ4WordBinding live);
}
