#pragma once
#include "strata/qwen35/native/small_prefill.hpp"

namespace strata::qwen35::native::detail {
// One256-column by32-token tile. Each token owns128 packed half2 words.
// Full F32 Q8 delta is retained; every padded token is explicitly zero.
struct alignas(16) HalfInputBlock {
    float delta[8][32];
    unsigned codes[32][128];
};
static_assert(sizeof(unsigned)==4,"half2 words require32-bit unsigned");
static_assert(sizeof(HalfInputBlock)==17408,"prepared half input block extent");
static_assert(alignof(HalfInputBlock)==16,"prepared half input vector alignment");
}
