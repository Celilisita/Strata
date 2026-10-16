// Minimal backend registration; descriptors are owned by the shared provider.
void native_mmvq_gathered(int type, const NativeTcTile* tiles, int tile_count,
    const void* q8, float* output, int n_in, int n_out, void* stream, int tile_rows) {
    if ((type != 12 && type != 14) || n_in <= 0 || n_in % 256 || n_out <= 0 ||
        (tile_rows != 1 && tile_rows != 2) || (tile_rows != 1 && type != 12) ||
        tile_count <= 0 || tile_count > 8192 || !tiles || !q8 || !output || !stream)
        throw std::invalid_argument("invalid gathered native MMVQ shape");
    const dim3 grid(unsigned(n_out), unsigned(tile_count), 2);
    const dim3 threads(WARP, WARPS);
    const auto st = static_cast<cudaStream_t>(stream);
    if (type == 12 && tile_rows == 2)
        native_mmvq_gathered_kernel<Q4KTraits, 2><<<dim3(unsigned((n_out + 1) / 2), unsigned(tile_count), 2), threads, 0, st>>>(
            tiles, static_cast<const Q81Block*>(q8), output, n_in, n_out);
    else if (type == 12)
        native_mmvq_gathered_kernel<Q4KTraits><<<grid, threads, 0, st>>>(
            tiles, static_cast<const Q81Block*>(q8), output, n_in, n_out);
    else
        native_mmvq_gathered_kernel<Q6KTraits><<<grid, threads, 0, st>>>(
            tiles, static_cast<const Q81Block*>(q8), output, n_in, n_out);
    launch_check();
}
