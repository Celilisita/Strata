#pragma once
#include "strata/qwen35/native/small_prefill.hpp"

namespace strata::qwen35::native::detail {
// A256-column x32-token code tile. Metadata and code words have contiguous
// token addresses; every padded token is explicitly encoded as zero.
struct alignas(16) VectorBlock {
    float delta[8][32];
    unsigned ds[8][32];
    int codes[64][32];
};
static_assert(sizeof(VectorBlock)==40*8*32,"blocked input extent");
}
