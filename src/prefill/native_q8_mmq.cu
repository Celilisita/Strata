#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/native_q6_accum_mmq.hpp"
#include "mmq.cuh"
#include <stdexcept>

namespace strata::prefill::mmq {
namespace {
// A warp copies one entire128-code native group. Four cells per warp keep
// block scheduling bounded; code words and scales retain their original bits.
__global__ void repack_q6_warp(const block_q8_1* __restrict__ input,
    block_q8_1_mmq* __restrict__ output, int cols, int rows) {
    const int lane = int(threadIdx.x)%32, warp = int(threadIdx.x)/32;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int cell = int(blockIdx.x)*16+warp+4*i;
        if (cell >= rows*(cols/128)) continue;
        const int token = cell%rows, kb = cell/rows;
        const auto* source = input+size_t(token)*(cols/32)+4*kb;
        reinterpret_cast<uint32_t*>(output[cell].qs)[lane] =
            reinterpret_cast<const uint32_t*>(source[lane/8].qs)[lane%8];
        if (lane < 4) output[cell].d4[lane] = __low2float(source[lane].ds);
    }
}
template<bool ScaleFloat>
__global__ void repack(const block_q8_1* input, block_q8_1_mmq* output, block_q8_1_mmq* residual, int cols, int rows) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= rows * (cols / QK8_1_MMQ)) return;
    const int token = cell % rows, kb = cell / rows;
    const auto* source = input + token * (cols / QK8_1) + 4 * kb;
    auto& destination = output[cell];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (ScaleFloat) destination.d4[j] = __low2float(source[j].ds);
        else {
            // MMVQ's affine minimum term uses the quantized-code sum, not
            // the original F32 sum stored in native Q8's upper half.
            int sum = 0;
#pragma unroll
            for (int k = 0; k < QK8_1; ++k) sum += int(source[j].qs[k]);
            const float full_sum = __low2float(source[j].ds) * sum;
            const half high = __float2half_rn(full_sum);
            destination.ds4[j] = make_half2(__low2half(source[j].ds), high);
            if (residual) residual[cell].ds4[j] = make_half2(__float2half_rn(0.0f),
                __float2half_rn(full_sum - __half2float(high)));
        }
        const auto* codes = reinterpret_cast<const uint32_t*>(source[j].qs);
        auto* dst = reinterpret_cast<uint32_t*>(destination.qs + j * QK8_1);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            dst[k] = codes[k];
            if (residual) reinterpret_cast<uint32_t*>(residual[cell].qs + j * QK8_1)[k] = 0;
        }
    }
}
}
void repack_q6_canonical(const void* source, void* packed, int width, int tokens, void* stream) {
    if (!source || !packed || !stream || width < 512 || width%512 || width > 16384 ||
        tokens < 1 || tokens > 8192)
        throw std::invalid_argument("invalid bounded canonical Q6 input conversion");
    const int cells = tokens*(width/128);
    auto* destination = static_cast<block_q8_1_mmq*>(packed);
    const auto st = static_cast<cudaStream_t>(stream);
    auto error = cudaMemsetAsync(destination+cells, 0, 128*sizeof(*destination), st);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    repack_q6_warp<<<(cells+15)/16,128,0,st>>>(static_cast<const block_q8_1*>(source), destination,width,tokens);
    error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void repack_native_q8(const void* input, void* output, int type, int cols, int rows, void* stream, void* sum_residual) {
    if (!input || !output || !stream || (type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K && type != GGML_TYPE_Q6_K) ||
        cols < 512 || cols % 512 || rows < 1 || rows > 8192 || (sum_residual && type != GGML_TYPE_Q4_K))
        throw std::invalid_argument("invalid canonical Q8 MMQ layout conversion");
    const int cells = rows * (cols / QK8_1_MMQ);
    const auto st = static_cast<cudaStream_t>(stream);
    auto* dst = static_cast<block_q8_1_mmq*>(output);
    auto* correction = static_cast<block_q8_1_mmq*>(sum_residual);
    const auto* src = static_cast<const block_q8_1*>(input);
    cudaError_t error = cudaMemsetAsync(dst + cells, 0, 128 * sizeof(*dst), st);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (correction) {
        error = cudaMemsetAsync(correction + cells, 0, 128 * sizeof(*correction), st);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
    // Q4_K/Q5_K affine loaders require DS4; only Q6_K uses D4.
    if (type == GGML_TYPE_Q6_K) repack<true><<<(cells + 127) / 128, 128, 0, st>>>(src, dst, correction, cols, rows);
    else repack<false><<<(cells + 127) / 128, 128, 0, st>>>(src, dst, correction, cols, rows);
    error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::prefill::mmq
