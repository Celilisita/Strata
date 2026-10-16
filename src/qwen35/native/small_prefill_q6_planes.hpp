#pragma once
#include "strata/qwen35/native/ffn.hpp"

namespace strata::qwen35::native {
// Lossless 64-row/K256 tiles: 192 bytes of codes per row, eight contiguous
// scale-pair planes, then the original half-delta plane. Packed=210B/block;
// optional signed-byte codes=274B/block with exactly the same integers.
struct Q6PlaneMatrix {
    const void *data=nullptr;
    int rows=0,columns=0;
    size_t bytes=0;
};
Q6PlaneMatrix prepare_q6_planes(Matrix,void *,cudaStream_t);
Q6PlaneMatrix prepare_q6_byte_planes(Matrix,void *,cudaStream_t);
inline size_t q6_byte_planes_bytes(Matrix raw){
    matrix_bytes(raw);
    if(raw.quant!=Quant::Q6K||raw.layout!=BlockLayout::Gguf||raw.rows%64)
        throw std::invalid_argument("byte planes require complete raw Q6 tiles");
    return size_t(raw.rows)*(raw.columns/256)*274;
}
void small_prefill_q6_planes_linear(Q6PlaneMatrix,const void *,float *,cudaStream_t);
}
