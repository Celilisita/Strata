#include "strata/qwen35/native/ffn.hpp"
#include <string>
#include <stdexcept>
namespace strata::qwen35::native {
void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
Capabilities capabilities(int device) {
    cudaDeviceProp p{};
    check_cuda(cudaGetDeviceProperties(&p, device), "query native device capabilities");
    if (p.major < 7) throw std::runtime_error("native FP16 Tensor Core path requires CC >= 7.0");
    int clusters = 0;
    check_cuda(cudaDeviceGetAttribute(&clusters, cudaDevAttrClusterLaunch, device), "query clusters");
    return {p.major, p.minor, p.multiProcessorCount, p.sharedMemPerBlockOptin,
            p.major >= 8, clusters != 0, p.major == 10};
}
}
