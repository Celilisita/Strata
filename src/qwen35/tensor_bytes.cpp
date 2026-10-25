#include "strata/qwen35/spec.hpp"
#include <limits>
#include <stdexcept>
namespace strata::qwen35 {
uint64_t tensor_bytes(const GgufFile &gguf, const TensorInfo &tensor) {
    uint64_t elements = 1;
    for (uint64_t dimension : tensor.shape) {
        if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension)
            throw std::runtime_error("invalid tensor dimensions: " + tensor.name);
        elements *= dimension;
    }
    int block = 0, bytes = 0;
    if (!block_geometry(tensor.type, block, bytes) || block <= 0 || bytes <= 0 ||
        elements % uint64_t(block) ||
        elements / uint64_t(block) > (std::numeric_limits<uint64_t>::max)() / uint64_t(bytes))
        throw std::runtime_error("unsupported tensor storage: " + tensor.name);
    const uint64_t size = elements / uint64_t(block) * uint64_t(bytes);
    if (gguf.data_start() > gguf.file_size() ||
        tensor.offset > gguf.file_size() - gguf.data_start() ||
        size > gguf.file_size() - gguf.data_start() - tensor.offset)
        throw std::runtime_error("tensor exceeds GGUF file: " + tensor.name);
    return size;
}
}
