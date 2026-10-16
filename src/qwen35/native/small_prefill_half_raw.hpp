#pragma once
#include "strata/qwen35/native/small_prefill.hpp"

namespace strata::qwen35::native::detail {
// Q4 affine input preserves rounded delta and original F32 sum in half bits.
// Q6 retains its existing full-F32 delta descriptor. Both extents match.
struct alignas(16) HalfAffineInputBlock {
    unsigned ds[8][32];
    unsigned codes[32][128];
};
static_assert(sizeof(HalfAffineInputBlock)==17408,"affine prepared input extent");
}
namespace strata::qwen35::native {
struct RawHalfScratch { void *data=nullptr;size_t bytes=0; };
void small_prefill_q6_raw_prefetch_linear(Matrix,const void *,float *,cudaStream_t,RawHalfScratch={});
inline unsigned half_raw_slices(Matrix weight,int processors){
    return weight.quant==Quant::Q6K && weight.columns>=512 &&
           (weight.rows+63)/64<2*processors?(weight.layout==BlockLayout::Q6Codes?3:2):1;
}
// Transient setup-free producer; Quant selects its metadata descriptor.
void small_prefill_half_raw_quantize(const float *,void *,Quant,int,int,cudaStream_t);
// Raw Q4/Q6 weight inventory. Only the input codes are prepared, on SM70.
void small_prefill_half_raw_linear(Matrix,const void *,float *,int,cudaStream_t,int,RawHalfScratch,bool allow_split=true);
}
