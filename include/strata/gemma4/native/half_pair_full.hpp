#pragma once
#include "half_row_pair.hpp"
#include <cstdint>
namespace strata::gemma4::native {
struct FullKPairBinding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct FullKPairLease {HalfRowPairLease operands;FullKPairBinding binding;};
// Independent unregistered1280 PP candidate. Same197 RN16 encoding; only the
// merged43008x1280x5376 reduction becomes one cuBLAS call, FP32accum/output.
void half_row_pair_full_project(primitive::Matrix,primitive::Matrix,
    const FullKPairLease&,FullKPairBinding live);
}
