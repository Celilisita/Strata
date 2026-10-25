#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/native_grouped_mmq.hpp"
#include "strata/prefill/native_code_sum_support.hpp"
#include "common.cuh"
#include "strata/prefill/native_q6_accum_mmq.hpp"
#include <cstdint>
#include <stdexcept>
// Separate backend instantiations prevent ODR collisions with ordinary DS4 MMQ.
// Every tile loader, K schedule, fixup and writeback stays in pinned ggml.
#define ggml_cuda_mmq_get_vec_dot strata_code_sum_get_vec_dot
#define mul_mat_q_process_tile strata_code_sum_process_tile
#define mul_mat_q strata_code_sum_mul_mat_q
#define mul_mat_q_stream_k_fixup strata_code_sum_fixup
#define launch_mul_mat_q strata_code_sum_launch
#define mul_mat_q_switch_J strata_code_sum_switch_J
#define mul_mat_q_case strata_code_sum_case
#include "native_code_sum_backend.cuh"
#include "native_mmq_precision.cuh"
#include "native_mmq_tile_choice.cuh"
#undef ggml_cuda_mmq_get_vec_dot
#undef mul_mat_q_process_tile
#undef mul_mat_q
#undef mul_mat_q_stream_k_fixup
#undef launch_mul_mat_q
#undef mul_mat_q_switch_J
#undef mul_mat_q_case
#include "native_q4_code_sum_dot.cuh"

namespace strata::prefill::mmq {
namespace {
bool code_sum_device(int& cc, size_t& shared) {
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    cc = dev.cc; shared = dev.smpbo; // Actual CUDA opt-in limit, not an assumed GPU name.
    const int arch = ggml_cuda_highest_compiled_arch(cc);
    if (cc == 1200 && arch == 1200) return true;
    return shared && (arch == GGML_CUDA_CC_VOLTA ||
        (cc == GGML_CUDA_CC_TURING && arch == GGML_CUDA_CC_TURING));
}
bool code_sum_config(const strata_code_sum_mmq_config& c, int type, int cc, size_t shared) {
    return c.type == type && !c.use_mma_data_layout(cc) &&
        c.I == native_code_sum_tile_rows && c.nthreads == native_code_sum_threads &&
        c.occupancy == 1 && c.K_vram == native_code_sum_k &&
        mmq_get_nbytes_shared(c, cc) <= shared;
}
__device__ constexpr bool code_sum_device_config() {
#if __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 750
    for (int type = GGML_TYPE_Q4_K; type <= GGML_TYPE_Q6_K; type += 2) {
        for (int fallback = 0; fallback < 2; ++fallback) {
            for (int j = 8; j <= 128; j += 8) {
                const auto c = ggml_cuda_mmq_get_config(ggml_type(type), j, bool(fallback) STRATA_NATIVE_MMQ_Q8);
                if (c.type == GGML_TYPE_COUNT) continue;
                if (c.use_mma_data_layout() || c.I != native_code_sum_tile_rows ||
                    c.nthreads != native_code_sum_threads || c.occupancy != 1 ||
                    c.K_vram != native_code_sum_k) return false;
            }
        }
    }
#endif
    return true;
}
template<bool D4>
__global__ void pack_code_sum(const block_q8_1* input, block_q8_1_mmq* output, int cols, int rows, const int32_t* ids = nullptr, int source_rows = 0) {
    static_assert(code_sum_device_config(), "private code-sum host/device DP4A contract");
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= rows * (cols / QK8_1_MMQ)) return;
    const int token = cell % rows, kb = cell / rows;
    const int source_row = ids ? ids[token] : token;
    if (ids && (source_row < 0 || source_row >= source_rows)) return;
    const auto* source = input + source_row * (cols / QK8_1) + 4 * kb;
    auto& dst = output[cell];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        if constexpr (D4) dst.d4[part] = __low2float(source[part].ds);
        else {
        int code_sum = 0;
#pragma unroll
        for (int k = 0; k < QK8_1; ++k) code_sum += int(source[part].qs[k]);
        auto* bits = reinterpret_cast<uint16_t*>(&dst.ds4[part]);
        bits[0] = __half_as_ushort(__low2half(source[part].ds));
        bits[1] = uint16_t(int16_t(code_sum));
        }
        const auto* codes = reinterpret_cast<const uint32_t*>(source[part].qs);
        auto* packed = reinterpret_cast<uint32_t*>(dst.qs + part * QK8_1);
#pragma unroll
        for (int k = 0; k < 8; ++k) packed[k] = codes[k];
    }
}
}
bool native_code_sum_dense_supported(int cols, int outputs, int tokens) {
    if (cols < 512 || cols % 512 || outputs < 1 || tokens < 1 || tokens > 8192) return false;
    int cc = 0; size_t shared = 0;
    if (!code_sum_device(cc, shared)) return false;
    if (cc == 1200) return true;
    const bool fallback = ggml_cuda_mmq_needs_fallback(outputs);
    bool legal = false;
    for (int j = 8; j <= 128; j += 8) {
        const auto c = ggml_cuda_mmq_get_config(GGML_TYPE_Q4_K, j, fallback, cc STRATA_NATIVE_MMQ_Q8);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > shared) continue;
        if (!code_sum_config(c, GGML_TYPE_Q4_K, cc, shared)) return false;
        legal = true;
    }
    return legal;
}
bool native_code_sum_grouped_supported(int type, int cols, int outputs) {
    if ((type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q6_K) ||
        cols < 512 || cols % 512 || outputs < 128 || outputs % native_code_sum_tile_rows) return false;
    int cc = 0; size_t shared = 0;
    if (!code_sum_device(cc, shared)) return false;
    if (cc == 1200) return true;
    const auto c = ggml_cuda_mmq_get_config(ggml_type(type), native_code_sum_group_columns,
        false, cc STRATA_NATIVE_MMQ_Q8);
    return code_sum_config(c, type, cc, shared);
}
void repack_native_q8_code_sum(const void* input, void* output, int cols, int rows, void* stream) {
    if (!input || !output || !stream || cols < 512 || cols % 512 || rows < 1 || rows > 8192)
        throw std::invalid_argument("invalid exact-code-sum MMQ layout");
    const int cells = rows * (cols / QK8_1_MMQ);
    const auto st = static_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaMemsetAsync(static_cast<block_q8_1_mmq*>(output) + cells, 0, 128 * sizeof(block_q8_1_mmq), st));
    pack_code_sum<false><<<(cells + 127) / 128, 128, 0, st>>>(static_cast<const block_q8_1*>(input), static_cast<block_q8_1_mmq*>(output), cols, rows);
    CUDA_CHECK(cudaGetLastError());
}
void repack_native_q8_grouped(const void* input, void* output, int type, int cols,
    int source_rows, int rows, const int32_t* source_ids, void* stream) {
    if ((type != 12 && type != 14) || !input || !output || !stream || !source_ids || cols < 512 || cols % 512 ||
        rows < 1 || rows > 65536 || source_rows < 1 || source_rows > 65536)
        throw std::invalid_argument("invalid bounded gathered code-sum layout");
    const int cells = rows * (cols / QK8_1_MMQ);
    const auto st = static_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaMemsetAsync(static_cast<block_q8_1_mmq*>(output) + cells, 0, 128 * sizeof(block_q8_1_mmq), st));
    if (type == GGML_TYPE_Q6_K)
        pack_code_sum<true><<<(cells + 127) / 128, 128, 0, st>>>(static_cast<const block_q8_1*>(input),
            static_cast<block_q8_1_mmq*>(output), cols, rows, source_ids, source_rows);
    else
        pack_code_sum<false><<<(cells + 127) / 128, 128, 0, st>>>(static_cast<const block_q8_1*>(input),
            static_cast<block_q8_1_mmq*>(output), cols, rows, source_ids, source_rows);
    CUDA_CHECK(cudaGetLastError());
}
void Context::run_code_sum(const Product& p, void* stream) {
    if (p.type != GGML_TYPE_Q4_K || p.n != 1 || !p.w || !p.xq || !p.dst || !stream ||
        p.w_cols < 512 || p.w_cols % 512 || p.w_rows < 1 || p.total_rows < 1 ||
        p.total_rows > 8192 || p.max_rows != p.total_rows || p.ld_dst != p.w_rows || p.ids || p.bounds)
        throw std::invalid_argument("unsupported exact-code-sum MMQ product");
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int compiled = ggml_cuda_highest_compiled_arch(cc);
    if (compiled == 1200) { run_q4_canonical(p,stream); return; }
    if (!native_code_sum_dense_supported(int(p.w_cols), int(p.w_rows), int(p.total_rows)))
        throw std::runtime_error("unsupported private DP4A code-sum MMQ device/shape");
    mmq_args a{};
    a.x = static_cast<const char*>(p.w); a.type_x = GGML_TYPE_Q4_K;
    a.y = static_cast<const int*>(p.xq); a.dst = p.dst;
    a.ncols_x = p.w_cols; a.nrows_x = p.w_rows; a.stride_row_x = p.w_cols / 256;
    a.ncols_dst = a.ncols_y = p.total_rows; a.nrows_dst = p.ld_dst;
    a.nchannels_x = a.nchannels_y = a.nsamples_x = a.nsamples_y = 1;
    a.stride_channel_x = int64_t(p.expert_bytes / ggml_type_size(GGML_TYPE_Q4_K));
    a.ncols_max = p.max_rows;
    select_native_mmq_tile(a, p.max_rows);
    auto& ctx = *static_cast<ggml_backend_cuda_context*>(ctx_);
    if (p.w_rows % 128) strata_code_sum_switch_J<GGML_TYPE_Q4_K, true STRATA_NATIVE_MMQ_Q8>(ctx, a, static_cast<cudaStream_t>(stream));
    else strata_code_sum_switch_J<GGML_TYPE_Q4_K, false STRATA_NATIVE_MMQ_Q8>(ctx, a, static_cast<cudaStream_t>(stream));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace strata::prefill::mmq

#include "native_grouped_mmq_tile.cuh"
