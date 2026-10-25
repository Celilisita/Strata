// Canonical pinned MMQ tile loader/product/writeback; only matrix address and
// route-row scheduling differ. No expert-weight gathering or F16 mirror.
#include "strata/prefill/native_code_sum_support.hpp"
namespace strata::prefill::mmq {
void native_grouped_canonical(int, const NativeMmqMatrix*, int, int, const int32_t*, const void*, float*,
    int, int, int, int, void*, const NativeMmqTile*, int, const NativeMmqEncodedInput*);
namespace {
template<ggml_type Type, int J, bool Active>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(Type, J, false STRATA_NATIVE_MMQ_Q8),
    ggml_cuda_mmq_get_occupancy(Type, J, false STRATA_NATIVE_MMQ_Q8))
__global__ void native_grouped_mmq_tile(const NativeMmqMatrix* matrices,
    const int32_t* output_ids, const int* packed, float* output,
    int rows, int cols, int outputs, const NativeMmqTile* tiles) {
    constexpr auto type = Type;
#if __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 750
    constexpr auto config = ggml_cuda_mmq_get_config(type, J, false STRATA_NATIVE_MMQ_Q8);
    static_assert(!config.use_mma_data_layout() && config.I == native_code_sum_tile_rows &&
        config.nthreads == native_code_sum_threads && config.occupancy == 1 &&
        config.K_vram == native_code_sum_k, "private grouped host/device DP4A contract");
#endif
    constexpr int I = ggml_cuda_mmq_get_I(type, J, false STRATA_NATIVE_MMQ_Q8);
    constexpr int warps = ggml_cuda_mmq_get_nthreads(type, J, false STRATA_NATIVE_MMQ_Q8) / 32;
    const auto tile = Active ? tiles[blockIdx.y] : NativeMmqTile{int(blockIdx.z), int(blockIdx.y)};
    const auto m = matrices[tile.matrix];
    const int count = m.end - m.begin, jt = tile.tile;
    if (jt * J >= count) return;
    const int out_tile = int(blockIdx.x), part = out_tile / (outputs / I);
    const int out_first = (out_tile % (outputs / I)) * I;
    extern __shared__ int ids[];
    for (int j0 = 0; j0 < J; j0 += warps * 32) {
        const int j = j0 + int(threadIdx.y) * 32 + int(threadIdx.x);
        if (j < J) ids[j] = jt * J + j < count ? output_ids[m.begin + jt * J + j] : 0;
    }
    __syncthreads();
    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
    strata_code_sum_process_tile<type, J, false, false STRATA_NATIVE_MMQ_Q8>(
        reinterpret_cast<const char*>(uintptr_t(part ? m.up : m.gate)),
        out_first * (cols / 256), packed + (m.begin + jt * J) * sz, ids,
        output + size_t(part) * rows * outputs + out_first, nullptr, nullptr,
        cols / 256, rows, outputs, outputs - out_first - 1,
        count - jt * J - 1, 0, cols / 256);
}
}
template<ggml_type Type>
void launch_native_grouped_mmq(const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* output_ids, const void* packed, float* output,
    int rows, int max_rows, int cols, int outputs, void* stream, const NativeMmqTile* tiles, int tile_count, const NativeMmqEncodedInput* encoded) {
    if ((parts != 1 && parts != 2) || !matrices || !output_ids || !packed || !output || !stream || groups < 1 || groups > 256 ||
        rows < 1 || rows > 65536 || max_rows < 1 || max_rows > 8192 || max_rows > rows ||
        cols < 512 || cols % 512 || outputs < 128 || outputs % 128)
        throw std::invalid_argument("invalid bounded grouped MMQ tile");
    if ((tiles && (tile_count < 1 || tile_count > 1280)) || (!tiles && tile_count))
        throw std::invalid_argument("invalid bounded active MMQ tiles");
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int compiled = ggml_cuda_highest_compiled_arch(cc);
    if (compiled == 1200) {
        native_grouped_canonical(Type,matrices,groups,parts,output_ids,packed,output,rows,max_rows,cols,outputs,stream,tiles,tile_count,encoded);
        return;
    }
    if (!native_code_sum_grouped_supported(int(Type), cols, outputs))
        throw std::runtime_error("unsupported private DP4A grouped MMQ device/shape");
    constexpr int J = native_code_sum_group_columns;
    const auto config = ggml_cuda_mmq_get_config(Type, J, false, cc STRATA_NATIVE_MMQ_Q8);
    const int shared = int(mmq_get_nbytes_shared(config, cc));
    if (tiles) {
        CUDA_SET_SHARED_MEMORY_LIMIT((native_grouped_mmq_tile<Type, J, true>), shared);
        const dim3 grid(unsigned(parts * outputs / config.I), unsigned(tile_count), 1);
        native_grouped_mmq_tile<Type, J, true><<<grid, dim3(32, unsigned(config.nthreads / 32)), shared,
            static_cast<cudaStream_t>(stream)>>>(matrices, output_ids, static_cast<const int*>(packed),
                output, rows, cols, outputs, tiles);
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT((native_grouped_mmq_tile<Type, J, false>), shared);
        const dim3 grid(unsigned(parts * outputs / config.I), unsigned((max_rows + J - 1) / J), unsigned(groups));
        native_grouped_mmq_tile<Type, J, false><<<grid, dim3(32, unsigned(config.nthreads / 32)), shared,
            static_cast<cudaStream_t>(stream)>>>(matrices, output_ids, static_cast<const int*>(packed),
                output, rows, cols, outputs, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
}
void native_grouped_mmq(int type, const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* output_ids, const void* packed, float* output,
    int rows, int max_rows, int cols, int outputs, void* stream, const NativeMmqTile* tiles, int tile_count, const NativeMmqEncodedInput* encoded) {
    // Q5 is routed directly to the explicit SM120 FP8 provider, never the
    // legacy Volta template or an implicit PackedQ6 canonical fallback.
    if (type == GGML_TYPE_Q5_K) {
        native_grouped_canonical(type,matrices,groups,parts,output_ids,packed,output,
            rows,max_rows,cols,outputs,stream,tiles,tile_count,encoded);
        return;
    }
    if (type == GGML_TYPE_Q4_K)
        launch_native_grouped_mmq<GGML_TYPE_Q4_K>(matrices, groups, parts, output_ids, packed, output,
            rows, max_rows, cols, outputs, stream, tiles, tile_count, encoded);
    else if (type == GGML_TYPE_Q6_K)
        launch_native_grouped_mmq<GGML_TYPE_Q6_K>(matrices, groups, parts, output_ids, packed, output,
            rows, max_rows, cols, outputs, stream, tiles, tile_count, encoded);
    else throw std::invalid_argument("unsupported grouped native MMQ quantization");
}
} // namespace strata::prefill::mmq
