#pragma once
#include "strata/prefill/moe_mmq.hpp"
namespace strata::prefill::mmq {
// Lossless warp-cooperative layout conversion; caller reserves the same guard.
void repack_q6_canonical(const void* source, void* packed, int width, int tokens, void* stream);
// Dense packed Q6 product with four-chain canonical F32 accumulation. Register
// matrix panels reuse inputs across rows; no scratch allocation. Returns actual
// launches; full tiles have no row/token guard.
int run_q4_canonical(const Product& product, void* stream);
int run_q6_canonical(const Product& product, void* stream);
}
