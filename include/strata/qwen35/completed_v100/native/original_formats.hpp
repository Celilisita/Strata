#pragma once
#include "strata/qwen35/completed_v100/native/quant.hpp"
namespace strata::completed_v100::qwen35::native {
constexpr size_t OriginalNvfp4HeaderBytes=16;
static_assert(sizeof(Matrix)==24,"Original format adapter must preserve Matrix ABI");
// Admission is explicit at operators that understand these formats. Older
// Q4/Q6-only kernels keep their original validators and reject unknown layouts.
inline size_t original_matrix_bytes(Matrix m,bool q6_codes=false,bool q6_scaled=false,bool q4_tiles=false) {
    if(m.quant==Quant::NVFP4) {
        if(!m.data||m.rows<1||m.columns<64||m.columns%64||m.layout!=BlockLayout::Nvfp4Scale||
           reinterpret_cast<uintptr_t>(m.data)%16)
            throw std::invalid_argument("Original NVFP4 matrix requires its owned scale descriptor");
        return OriginalNvfp4HeaderBytes+size_t(m.rows)*(m.columns/64)*36;
    }
    if(m.quant==Quant::Q8) {
        if(!m.data||m.rows<1||m.columns<32||m.columns%32||m.layout!=BlockLayout::Gguf)
            throw std::invalid_argument("Original Q8 matrix requires raw GGUF block layout");
        return size_t(m.rows)*(m.columns/32)*34;
    }
    return matrix_bytes(m,q6_codes,q6_scaled,q4_tiles);
}
inline bool original_projection(Matrix matrix){return matrix.quant==Quant::NVFP4||matrix.quant==Quant::Q8;}
}
