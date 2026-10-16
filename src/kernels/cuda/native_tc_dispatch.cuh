// Backend registration; each product reads the canonical quantized directory.
void native_tc_grouped(int type, const NativeTcTile* tiles, int tile_count,
    const void* q8, float* output, int n_in, int n_out, void* stream) {
    const int max_k = type == 12 ? Q4KTraits::BPI * 256 : Q6KTraits::BPI * 256;
    if ((type != 12 && type != 14) || n_in <= 0 || n_in % 256 || n_in > max_k ||
        n_out <= 0 || n_out % 8 || tile_count <= 0 || !tiles || !q8 || !output || !stream)
        throw std::invalid_argument("invalid native TC grouped shape");
    const dim3 grid(unsigned(n_out / 8), unsigned(tile_count));
    const auto s = static_cast<cudaStream_t>(stream);
    if (type == 12)
        native_tc_grouped_kernel<TcQ4><<<grid, 256, 0, s>>>(tiles, static_cast<const Q81Block*>(q8), output, n_in, n_out);
    else
        native_tc_grouped_kernel<TcQ6><<<grid, 256, 0, s>>>(tiles, static_cast<const Q81Block*>(q8), output, n_in, n_out);
    launch_check();
}
void native_tc_configure() {
    const auto a = cudaFuncSetAttribute(native_tc_grouped_kernel<TcQ4>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    const auto b = cudaFuncSetAttribute(native_tc_grouped_kernel<TcQ6>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    if (a != cudaSuccess || b != cudaSuccess)
        throw std::runtime_error("native TC shared-memory configuration failed");
}
