#include "strata/runtime/capacity_storage.hpp"
#pragma once
#include "strata/qwen35/native/weights.hpp"
namespace strata::gemma4::native {
namespace primitive=strata::qwen35::native;
// Immutable model inventory choice. GGUF remains the default; changing this
// storage policy requires a new Model and new graphs, not a hot-path switch.
enum class Q6DownStorage { Gguf, Aligned224 };
inline bool is_aligned_q6_down(primitive::Matrix matrix){
    return matrix.layout==primitive::BlockLayout::Q6Aligned224;
}
// Accept original Gemma Q4/Q6 matrices, or the single admitted native Down
// geometry. Other providers still reject the native layout by default.
inline size_t q6_down_weight_bytes(primitive::Matrix matrix){
    if(matrix.quant!=primitive::Quant::Q4K&&matrix.quant!=primitive::Quant::Q6K)
        throw std::invalid_argument("Gemma weight quantization");
    if(is_aligned_q6_down(matrix)){
        if(matrix.rows!=5376||matrix.columns!=21504)
            throw std::invalid_argument("Gemma native Q6 Down extent");
        if (!matrix.data || matrix.rows <= 0 || matrix.columns <= 0 || matrix.columns % 256)
            throw std::invalid_argument("invalid native block matrix");
        if (matrix.quant != primitive::Quant::Q6K || reinterpret_cast<uintptr_t>(matrix.data) % 32)
            throw std::invalid_argument("Q6 aligned layout requires an explicit compatible operator");
        return size_t(matrix.rows) * size_t(matrix.columns / 256) * 224;
    }
    if(matrix.layout!=primitive::BlockLayout::Gguf)
        throw std::invalid_argument("Gemma weight layout");
    return primitive::matrix_bytes(matrix);
}
// raw must be a unique allocation of owner with no published aliases/graphs.
// The returned matrix is the only published inventory. A completed pack joins
// stream before release(raw); failures leave unpublished storage owned by owner
// for normal owner cleanup. Call only during cold inventory construction.
primitive::Matrix replace_owned_q6_down(primitive::Matrix raw,
    strata::runtime::CapacityStorage& owner,cudaStream_t stream);
}
