// Scheduling views only; the quantized MMA product has one implementation.
template<bool Dense> struct NativeTcView;
template<> struct NativeTcView<false> {
    const NativeTcTile& tile;
    __device__ NativeTcView(const NativeTcTile* tiles, const void*, int, NativeTcPacked)
        : tile(tiles[blockIdx.y]) {}
    __device__ const void* weight() const { return reinterpret_cast<const void*>(uintptr_t(tile.address)); }
    __device__ int input_row(int token) const { return tile.input_rows[token]; }
    __device__ int output_row(int token) const { return tile.output_rows[token]; }
};
template<> struct NativeTcView<true> {
    NativeTcPacked packed;
    int first, count;
    __device__ NativeTcView(const NativeTcTile*, const void*, int columns, NativeTcPacked p)
        : packed(p), first(int(blockIdx.y) * 16), count(columns) {}
    __device__ const void* weight() const { return nullptr; }
    __device__ int input_row(int token) const { return first + token < count ? first + token : -1; }
    __device__ int output_row(int token) const { return input_row(token); }
};
