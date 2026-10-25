#include "strata/qwen35/native/gdn_prefix.hpp"
#include "strata/qwen35/native/ffn.hpp"

namespace strata::qwen35::native {
namespace {
__global__ void retain_history(const float *input, const float *history,
                                GdnPrefix prefix, int channels, int tokens) {
    const int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= 3 * channels * (tokens - 1)) return;
    const int slot = index / (3 * channels), element = index % (3 * channels);
    const int channel = element / 3, lag = element % 3;
    const int source = slot + 1 - 3 + lag;
    prefix.history[slot][element] = source < 0 ? history[3 * channel + source + 3]
        : input[size_t(source) * channels + channel];
}
}
void conv_prefix_history(const float *input, const float *history, GdnPrefix prefix,
                         int channels, int tokens, cudaStream_t stream) {
    if (!input || !history || channels < 1 || channels > 65536 || tokens < 2 || tokens > 4)
        throw std::invalid_argument("invalid retained convolution history geometry");
    for(int slot=0;slot<tokens-1;++slot){
        const auto *p=prefix.history[slot];
        if(!p||p==history||p==input)throw std::invalid_argument("invalid retained convolution history lease");
        for(int earlier=0;earlier<slot;++earlier)if(p==prefix.history[earlier])
            throw std::invalid_argument("aliased retained convolution histories");
    }
    const int count = 3 * channels * (tokens - 1);
    retain_history<<<(count + 255) / 256, 256, 0, stream>>>(input, history, prefix, channels, tokens);
    check_cuda(cudaGetLastError(), "native retained convolution prefix history");
}
}
