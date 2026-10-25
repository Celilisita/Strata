#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::completed_v100::qwen35::native {
// GGUF block format identifiers, not a device execution policy.
enum class Quant : uint32_t { Q8 = 8, Q4K = 12, Q6K = 14, NVFP4 = 40 };
enum class BlockLayout { Gguf, Q4Pairs, Q6Codes, Q6Scaled, Q4Tiles, Nvfp4Scale, Q6Aligned224 };
// Original210-byte Q6K payload followed by fourteen zero bytes. Consumers must
// opt in explicitly; appending this value preserves every previous enum value.
constexpr size_t Q6AlignedBlockBytes = 224;
// Q6Codes stores all256-byte code blocks before all18-byte metadata blocks.
// Metadata preserves sixteen signed scale bytes and the original half delta.
constexpr size_t Q6CodesBytes = 256, Q6CodesMetadataBytes = 18;
constexpr size_t Q6CodesBlockBytes = Q6CodesBytes + Q6CodesMetadataBytes;
// Exact scaled-half plane, one-bit positive integer residual, original delta.
constexpr size_t Q6ScaledBlockBytes = 512 + 32 + 2;
// Owned transient activation block: half scale, padding, 32 codes.
// This is not a GGUF weight format or a persistent model conversion.
constexpr size_t VectorActivationBytes = 36;
struct Matrix {
    const void *data = nullptr;
    Quant quant = Quant::Q4K;
    int rows = 0, columns = 0;
    BlockLayout layout = BlockLayout::Gguf;
};
inline size_t block_bytes(Quant type) {
    switch (type) {
        case Quant::Q8: return 34;
        case Quant::Q4K: return 144;
        case Quant::Q6K: return 210;
    }
    throw std::invalid_argument("unsupported native quant format");
}
inline size_t matrix_bytes(const Matrix &m, bool allow_q6_codes = false, bool allow_q6_scaled = false, bool allow_q4_tiles = false, bool allow_q6_aligned = false) {
    if (!m.data || m.rows <= 0 || m.columns <= 0 || m.columns % (m.quant == Quant::Q8 ? 32 : 256))
        throw std::invalid_argument("invalid native block matrix");
    if (m.layout == BlockLayout::Q6Aligned224) {
        if (!allow_q6_aligned || m.quant != Quant::Q6K || reinterpret_cast<uintptr_t>(m.data)%32)
            throw std::invalid_argument("Q6 aligned layout requires an explicit compatible operator");
        return size_t(m.rows) * size_t(m.columns / 256) * Q6AlignedBlockBytes;
    }
    if(m.layout==BlockLayout::Q4Tiles){
        if(!allow_q4_tiles||m.quant!=Quant::Q4K||m.rows%64||reinterpret_cast<uintptr_t>(m.data)%16)
            throw std::invalid_argument("Q4 tiled layout requires an explicit compatible operator");
        return size_t(m.rows)*size_t(m.columns/256)*144;
    }
    if (m.layout == BlockLayout::Q6Codes) {
        if (!allow_q6_codes || m.quant != Quant::Q6K)
            throw std::invalid_argument("Q6 code layout requires an explicit compatible operator");
        return size_t(m.rows) * size_t(m.columns / 256) * Q6CodesBlockBytes;
    }
    if (m.layout == BlockLayout::Q6Scaled) {
        if (!allow_q6_scaled || m.quant != Quant::Q6K)
            throw std::invalid_argument("Q6 scaled layout requires an explicit compatible operator");
        return size_t(m.rows) * size_t(m.columns / 256) * Q6ScaledBlockBytes;
    }
    if (m.layout != BlockLayout::Gguf && (m.layout != BlockLayout::Q4Pairs || m.quant != Quant::Q4K))
        throw std::invalid_argument("invalid native weight block layout");
    return size_t(m.rows) * size_t(m.columns / (m.quant == Quant::Q8 ? 32 : 256)) * block_bytes(m.quant);
}
} // namespace strata::completed_v100::qwen35::native
