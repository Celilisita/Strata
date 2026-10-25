#pragma once
#include "strata/qwen35/native/quant.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::gemma4::native {
namespace primitive=strata::qwen35::native;
// Independent precision: the first two residual-Q8 planes of179, not Single40
// or a byte prefix of120B. F32 scales and signed integer group sums are retained.
struct alignas(4) TwoPlaneQ8Block {
    float d[2];int sum[2];signed char code[2][32];
};
static_assert(sizeof(int)==4&&sizeof(TwoPlaneQ8Block)==80);
static_assert(offsetof(TwoPlaneQ8Block,sum)==8&&offsetof(TwoPlaneQ8Block,code)==16);
struct TwoPlaneQ8Binding {
    const void* owner=nullptr;
    std::uint64_t epoch=0;
    cudaStream_t stream=nullptr;
};
struct TwoPlaneQ8Lease {
    const TwoPlaneQ8Block* data=nullptr;
    size_t bytes=0;
    int columns=0,tokens=0;
    const float* source=nullptr;
    void* arena=nullptr;
    size_t arena_bytes=0;
    TwoPlaneQ8Binding binding;
};
// T1 QKV/head only: actual hidden width5376; producer and all consumers use the
// same ordered stream. Owner increments epoch whenever this arena is reused.
// Source, weights and lease are immutable until all consumers complete. The
// lease expires on arena reuse; caller supplies the live binding on every use.
size_t two_plane_q8_t1_bytes(int columns);
TwoPlaneQ8Lease prepare_two_plane_q8_t1(const float*,void* arena,size_t arena_bytes,
    int columns,TwoPlaneQ8Binding);
// Explicit unregistered API. Raw GGUF Q4/Q6 only; fixed Q/K/V/head row extents.
// Does not allocate, synchronize, copy weights or silently fall back.
void linear_two_plane_q8_t1(primitive::Matrix,const TwoPlaneQ8Lease&,float*,TwoPlaneQ8Binding);
}
