#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "codec.cuh"
#include "math.cuh"

namespace strata::completed_v100::qwen35::native {
namespace {
template<bool Interleaved=false>
__global__ void unpack(Matrix a, Matrix b, half *output, bool pair) {
    const size_t block = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const size_t count = size_t(a.rows) * a.columns / 256;
    if (block >= count * (pair ? 2 : 1)) return;
    const Matrix source = block < count ? a : b;
    const int stride = source.quant == Quant::Q4K ? 144 : 210;
    const auto *p = static_cast<const unsigned char *>(source.data) + (block % count) * stride;
    size_t target=block;
    if constexpr(Interleaved){
        const unsigned within=unsigned(block%count),blocks=unsigned(a.columns)/256;
        target=(2*(within/blocks)+size_t(block>=count))*blocks+within%blocks;
    }
    half *destination = output + target * 256;
    if (source.quant == Quant::Q4K) {
        const auto own = lane < 8 ? q4_coefficients(p, lane, source.layout) : Q4Coefficients{0, 0};
        // Eight metadata producers feed every column lane. Each packed byte is
        // fetched once and supplies the low/high pair of logical 32-value groups.
        for (int plane = 0; plane < 4; ++plane) {
            const unsigned char code = p[16 + plane * 32 + lane];
            for (int nibble = 0; nibble < 2; ++nibble) {
                const int group = plane * 2 + nibble;
                const float scale = __shfl_sync(0xffffffff, own.scale, group);
                const float minimum = __shfl_sync(0xffffffff, own.minimum, group);
                const float value = __fsub_rn(__fmul_rn(scale, float((code >> (nibble * 4)) & 15)), minimum);
                destination[group * 32 + lane] = __float2half_rn(value);
            }
        }
    } else {
        const float d = __half2float(*reinterpret_cast<const half *>(p + 208));
        const float own = lane < 16 ? __fmul_rn(d, float(reinterpret_cast<const signed char *>(p + 192)[lane])) : 0;
        for (int section = 0; section < 2; ++section) {
            const unsigned char high = p[128 + section * 32 + lane];
            for (int group = 0; group < 4; ++group) {
                const unsigned char low = p[section * 64 + (group & 1) * 32 + lane];
                const int code = (((low >> ((group / 2) * 4)) & 15) | (((high >> (group * 2)) & 3) << 4)) - 32;
                const float scale = __shfl_sync(0xffffffff, own, section * 8 + group * 2 + lane / 16);
                destination[section * 128 + group * 32 + lane] = __float2half_rn(__fmul_rn(scale, float(code)));
            }
        }
    }
}
__global__ void convert(const float *input, half *output, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half_rn(input[i]);
}
__global__ void activate(const float *gate, const float *up, float *output, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        // Match the established CUDA activation approximation explicitly;
        // quantization and Tensor Core code keep their own rounding policy.
        output[i] = swiglu_value(gate[i], up[i]);
    }
}
}
void decode_pair(Matrix a, Matrix b, void *output, cudaStream_t stream) {
    const size_t count = size_t(a.rows) * a.columns * 2;
    unpack<><<<unsigned((count / 256 + 3) / 4), 128, 0, stream>>>(a, b, static_cast<half *>(output), true);
    check_cuda(cudaGetLastError(), "native paired weight decode");
}
void decode_pair_interleaved(Matrix a,Matrix b,void *output,cudaStream_t stream){
    matrix_bytes(a);matrix_bytes(b);
    if(a.rows!=b.rows||a.columns!=b.columns||!output)
        throw std::invalid_argument("interleaved weight geometry changed");
    const size_t count=size_t(a.rows)*a.columns*2;
    if(count/512>0x7fffffffu)throw std::invalid_argument("interleaved row/block extent exceeds32-bit index");
    unpack<true><<<unsigned((count/256+3)/4),128,0,stream>>>(a,b,static_cast<half *>(output),true);
    check_cuda(cudaGetLastError(),"Strata interleaved Gate/Up decode");
}
void decode_linear(Matrix a, void *output, cudaStream_t stream) {
    const size_t count = size_t(a.rows) * a.columns;
    unpack<><<<unsigned((count / 256 + 3) / 4), 128, 0, stream>>>(a, a, static_cast<half *>(output), false);
    check_cuda(cudaGetLastError(), "native weight decode");
}
void to_half(const float *input, void *output, size_t count, cudaStream_t stream) {
    convert<<<unsigned((count + 255) / 256), 256, 0, stream>>>(input, static_cast<half *>(output), count);
    check_cuda(cudaGetLastError(), "native activation conversion");
}
void swiglu(const float *gate, const float *up, float *output, size_t count, cudaStream_t stream) {
    activate<<<unsigned((count + 255) / 256), 256, 0, stream>>>(gate, up, output, count);
    check_cuda(cudaGetLastError(), "native SwiGLU");
}
} // namespace strata::completed_v100::qwen35::native
