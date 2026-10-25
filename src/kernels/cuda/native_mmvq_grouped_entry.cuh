// Diagnostic count observes dispatch only; no route/cache/model authority.
static thread_local unsigned long long tg_down_row4_submissions = 0;
extern "C" unsigned long long strata_tg_down_row4_submissions() { return tg_down_row4_submissions; }
// Probe-only resource observation. Production dispatch never calls CUDA queries here.
template<typename F> int tg_down_row4_resource_values(int* values) {
    cudaFuncAttributes attributes{}; int active = 0;
    auto error = cudaFuncGetAttributes(&attributes, native_mmvq_grouped_down_row4_kernel<F>);
    if (error != cudaSuccess) return int(error);
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, native_mmvq_grouped_down_row4_kernel<F>, 4 * WARP, 0);
    if (error != cudaSuccess) return int(error);
    values[0] = attributes.binaryVersion; values[1] = attributes.numRegs;
    values[2] = int(attributes.localSizeBytes); values[3] = int(attributes.sharedSizeBytes);
    values[4] = active; return 0;
}
extern "C" int strata_tg_down_row4_resources(int type, int* values) {
    if (!values) return -1;
    if (type == 12) return tg_down_row4_resource_values<Q4KTraits>(values);
    if (type == 13) return tg_down_row4_resource_values<Q5KTraits>(values);
    if (type == 14) return tg_down_row4_resource_values<Q6KTraits>(values);
    return -1;
}
// Host submissions only, not graph replays or decoded-token counts.
static thread_local unsigned long long tg_gu_row2_submissions = 0;
extern "C" unsigned long long strata_tg_gu_row2_submissions() { return tg_gu_row2_submissions; }
// Probe-only resource query; production dispatch performs no resource CUDA calls.
extern "C" int strata_tg_gu_row2_resources(int* values) {
    if (!values) return -1;
    cudaFuncAttributes attributes{}; int active = 0;
    auto error = cudaFuncGetAttributes(&attributes, native_mmvq_grouped_gu_row2_kernel);
    if (error != cudaSuccess) return int(error);
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, native_mmvq_grouped_gu_row2_kernel, 4 * WARP, 0);
    if (error != cudaSuccess) return int(error);
    values[0] = attributes.binaryVersion; values[1] = attributes.numRegs;
    values[2] = int(attributes.localSizeBytes); values[3] = int(attributes.sharedSizeBytes);
    values[4] = active; return 0;
}
void native_mmvq_grouped(int ggml_type, const unsigned long long* addresses,
                         const int32_t* input_rows, const void* x_q8_1, float* y,
                         int n_in, int n_out, int entries, void* stream) {
    if ((ggml_type != 12 && ggml_type != 13 && ggml_type != 14) || n_in <= 0 || n_in % 256 ||
        n_out <= 0 || entries <= 0 || entries > 131072 || !addresses || !input_rows ||
        !x_q8_1 || !y || !stream)
        throw std::invalid_argument("invalid grouped native MMVQ shape or type");
    const char* row2_setting = std::getenv("STRATA_QWEN35_TG_GU_ROW2");
    if (row2_setting && !((row2_setting[0] == '0' || row2_setting[0] == '1') && row2_setting[1] == '\0'))
        throw std::invalid_argument("TG_GU_ROW2 requires 0 or 1");
    const char* row4_setting = std::getenv("STRATA_QWEN35_TG_DOWN_ROW4");
    if (row4_setting && !((row4_setting[0] == '0' || row4_setting[0] == '1') && row4_setting[1] == '\0'))
        throw std::invalid_argument("TG_DOWN_ROW4 requires 0 or 1");
    if (row4_setting && row4_setting[0] == '1' && n_in == 512 && n_out == 2048 && entries == 8) {
        const dim3 row4_grid{unsigned(n_out / 4), unsigned(entries)};
        const dim3 row4_threads(WARP, 4);
        const auto st = static_cast<cudaStream_t>(stream);
        if (ggml_type == 12)
            native_mmvq_grouped_down_row4_kernel<Q4KTraits><<<row4_grid, row4_threads, 0, st>>>(
                addresses, input_rows, static_cast<const Q81Block*>(x_q8_1), y, n_out);
        else if (ggml_type == 13)
            native_mmvq_grouped_down_row4_kernel<Q5KTraits><<<row4_grid, row4_threads, 0, st>>>(
                addresses, input_rows, static_cast<const Q81Block*>(x_q8_1), y, n_out);
        else
            native_mmvq_grouped_down_row4_kernel<Q6KTraits><<<row4_grid, row4_threads, 0, st>>>(
                addresses, input_rows, static_cast<const Q81Block*>(x_q8_1), y, n_out);
        launch_check();
        ++tg_down_row4_submissions;
        return;
    }
    if (row2_setting && row2_setting[0] == '1' && ggml_type == 12 &&
        n_in == 2048 && n_out == 512 && entries == 16) {
        const auto st = static_cast<cudaStream_t>(stream);
        native_mmvq_grouped_gu_row2_kernel<<<dim3(256, 16), dim3(WARP, 4), 0, st>>>(
            addresses, input_rows, static_cast<const Q81Block*>(x_q8_1), y);
        launch_check(); ++tg_gu_row2_submissions; return;
    }
    const dim3 grid{unsigned(n_out), unsigned(entries < 16384 ? entries : 16384), unsigned((entries + 16383) / 16384)};
    const dim3 threads(WARP, WARPS);
    const auto st = static_cast<cudaStream_t>(stream);
    if (ggml_type == 12)
        native_mmvq_grouped_kernel<Q4KTraits><<<grid, threads, 0, st>>>(
            addresses, input_rows, static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out, entries);
    else if (ggml_type == 13)
        native_mmvq_grouped_kernel<Q5KTraits><<<grid, threads, 0, st>>>(
            addresses, input_rows, static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out, entries);
    else
        native_mmvq_grouped_kernel<Q6KTraits><<<grid, threads, 0, st>>>(
            addresses, input_rows, static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out, entries);
    launch_check();
}

// Host submissions, including captures; graph replays never increment this count.
static thread_local unsigned long long tg_dense_q4_row2_submissions = 0;
extern "C" unsigned long long strata_tg_dense_q4_row2_submissions() { return tg_dense_q4_row2_submissions; }
// Probe-only: Runtime dispatch below never calls these CUDA resource queries.
extern "C" int strata_tg_dense_q4_row2_resources(int* values) {
    if (!values) return -1;
    cudaFuncAttributes attributes{}; int active = 0;
    auto error = cudaFuncGetAttributes(&attributes, native_q4_k_dense_row2_kernel);
    if (error != cudaSuccess) return int(error);
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, native_q4_k_dense_row2_kernel, 4 * WARP, 0);
    if (error != cudaSuccess) return int(error);
    values[0] = attributes.binaryVersion; values[1] = attributes.numRegs;
    values[2] = int(attributes.localSizeBytes); values[3] = int(attributes.sharedSizeBytes);
    values[4] = active; return 0;
}
void native_q4_k_dense_row2(const void* weights, const void* input_q8, float* output,
    int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_in != 2048 || ncols != 1 || (n_out != 512 && n_out != 4096 && n_out != 8192))
        throw std::invalid_argument("dense Q4 row2 requires K2048 N512/4096/8192 ncols1");
    validate_pointer(weights); validate_pointer(input_q8); validate_pointer(output);
    validate_stream(stream);
    native_q4_k_dense_row2_kernel<<<unsigned(n_out / 2), dim3(WARP, 4), 0,
        static_cast<cudaStream_t>(stream)>>>(static_cast<const Q4KBlock*>(weights),
        static_cast<const Q81Block*>(input_q8), output);
    launch_check(); ++tg_dense_q4_row2_submissions;
}

#include "native_mmvq_gu_cluster_entry.cuh"

#include "native_mmvq_quant_down_pair_entry.cuh"
