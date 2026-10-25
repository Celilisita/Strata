#pragma once
#include "strata/gemma4/native/ffn.hpp"
#include <cstddef>

namespace strata::gemma4::native {
// One independently quantized activation plane. GGUF weight ownership and
// layout are unchanged. This type must never alias a three-residual view.
struct alignas(4) SingleQ8Block {
    float d;
    int sum;
    signed char code[32];
};
static_assert(sizeof(SingleQ8Block)==40);
static_assert(alignof(SingleQ8Block)==4&&offsetof(SingleQ8Block,code)==8);

struct SingleQ8View {
    const SingleQ8Block* data=nullptr;
    size_t bytes=0;
    int columns=0;
    int tokens=0;
};

SingleQ8View prepare_single_q8(const float*,void* arena,size_t arena_bytes,
    int columns,int tokens,cudaStream_t);
void linear_single_q8(primitive::Matrix,SingleQ8View,float*,cudaStream_t);
void pair_single_q8(primitive::Matrix,primitive::Matrix,SingleQ8View,
    float* gate,float* up,float* activation,cudaStream_t);
}
