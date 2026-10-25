#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <stdexcept>
#include <string>

namespace strata::completed_v100::qwen35::native {
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
} // namespace strata::completed_v100::qwen35::native
