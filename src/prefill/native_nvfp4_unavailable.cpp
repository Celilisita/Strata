#include "strata/prefill/native_grouped_nvfp4.hpp"
#include <stdexcept>

namespace strata::prefill::mmq {
namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error("NVFP4 grouped CUDA provider was not compiled for SM120");
}
}
size_t native_nvfp4_input_bytes(int rows, int columns) {
    if (rows < 1 || rows > 65536 || columns < 64 || columns > 8192 || columns % 64)
        throw std::invalid_argument("bounded NVFP4 input geometry");
    return size_t(rows) * (columns / 2 + columns / 16 + 4);
}
NativeNvfp4Input encode_native_nvfp4(const float*, void*, size_t, int, int,
                                    const int32_t*, void*) {
    unavailable();
}
NativeNvfp4Input encode_native_nvfp4_masked(const float*, void*, size_t, int, int,
    const int32_t*, const unsigned long long*, void*) {
    unavailable();
}
void native_grouped_nvfp4(const NativeNvfp4Matrix*, int, int, const int32_t*,
    float*, int, int, int, const NativeMmqTile*, int, int, NativeNvfp4Input, void*) {
    unavailable();
}
NativeNvfp4Resources native_nvfp4_resources(int) {
    unavailable();
}
} // namespace strata::prefill::mmq
