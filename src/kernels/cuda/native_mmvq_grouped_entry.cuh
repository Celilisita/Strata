void native_mmvq_grouped(int ggml_type, const unsigned long long* addresses,
                         const int32_t* input_rows, const void* x_q8_1, float* y,
                         int n_in, int n_out, int entries, void* stream) {
    if ((ggml_type != 12 && ggml_type != 13 && ggml_type != 14) || n_in <= 0 || n_in % 256 ||
        n_out <= 0 || entries <= 0 || entries > 131072 || !addresses || !input_rows ||
        !x_q8_1 || !y || !stream)
        throw std::invalid_argument("invalid grouped native MMVQ shape or type");
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
