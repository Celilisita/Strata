// Included after canonical MMVQ dispatch, inside strata::kernels.
namespace {
template<typename F, int Columns>
void launch_multi_bulk(const void* weights, const void* input, float* output,
    int n_in, int n_out, int tiles, cudaStream_t stream) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(input);
    const dim3 threads(WARP, WARPS);
    if constexpr (Columns == 8) {
        if (n_in / F::DIV < F::BPI) {
            const unsigned rows = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
            native_mmvq_bulk_kernel<F, Columns, WARPS, WARPS, true>
                <<<dim3(rows, unsigned(tiles)), threads, 0, stream>>>(w, x, output, n_in, n_out);
            return;
        }
    }
    native_mmvq_bulk_kernel<F, Columns, WARPS, 1, true>
        <<<dim3(unsigned(n_out), unsigned(tiles)), threads, 0, stream>>>(w, x, output, n_in, n_out);
}
template<typename F, int Rows>
void launch_bulk_rows(const void* weights, const void* input, float* output,
    int n_in, int n_out, int columns, cudaStream_t stream) {
    native_mmvq_bulk_kernel<F, 8, WARPS, Rows, true, true>
        <<<dim3(unsigned((n_out + Rows - 1) / Rows), unsigned(columns / 8)), dim3(WARP, WARPS), 0, stream>>>(
            static_cast<const typename F::Block*>(weights), static_cast<const Q81Block*>(input), output, n_in, n_out);
}
template<typename F>
void launch_bulk_columns(const void* weights, const void* input, float* output,
    int n_in, int n_out, int columns, int width, int rows, cudaStream_t stream) {
    if (width == 4) launch_multi_bulk<F, 4>(weights, input, output, n_in, n_out, columns / 4, stream);
    else if (width == 32) launch_multi_bulk<F, 32>(weights, input, output, n_in, n_out, columns / 32, stream);
    else if (width == 16) launch_multi_bulk<F, 16>(weights, input, output, n_in, n_out, columns / 16, stream);
    else if (rows == 4) launch_bulk_rows<F, 4>(weights, input, output, n_in, n_out, columns, stream);
    else if (rows == 2) launch_bulk_rows<F, 2>(weights, input, output, n_in, n_out, columns, stream);
    else launch_multi_bulk<F, 8>(weights, input, output, n_in, n_out, columns / 8, stream);
}
} // namespace

bool native_mmvq_bulk_supported(int type) noexcept {
    return type == 2 || type == 6 || type == 8 || type == 20 || type == 11 ||
        type == 12 || type == 13 || type == 14 || type == 23 || type == 42;
}

std::size_t native_mmvq_bulk(int type, const void* weights, const void* input,
    float* output, int n_in, int n_out, int columns, void* stream, int tile_columns, int tile_rows) {
    if (!native_mmvq_bulk_supported(type) || columns < 8 || columns > 8192 || !g_multi_exact ||
        (tile_columns != 4 && tile_columns != 8 && tile_columns != 16 && tile_columns != 32) ||
        (tile_rows != 1 && tile_rows != 2 && tile_rows != 4) || (tile_rows != 1 && tile_columns != 8))
        throw std::invalid_argument("bulk MMVQ requires supported type, exact layout and8..8192 columns");
    // This validates the actual quant block dimensions, independently of dispatch.
    native_mmvq_weight_bytes(type, n_in, n_out);
    validate_pointer(weights); validate_pointer(input); validate_pointer(output);
    validate_stream(stream);
    const auto st = static_cast<cudaStream_t>(stream);
    const int width = columns >= tile_columns ? tile_columns : 8;
    switch (type) {
    case 2: launch_bulk_columns<SmallTraits<Q40Block, 4>>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 6: launch_bulk_columns<SmallTraits<Q50Block, 4>>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 8: launch_bulk_columns<SmallTraits<Q80Block, 8>>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 20: launch_bulk_columns<SmallTraits<IQ4NLBlock, 4>>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 11: launch_bulk_columns<Q3KTraits>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 12: launch_bulk_columns<Q4KTraits>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 13: launch_bulk_columns<Q5KTraits>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 14: launch_bulk_columns<Q6KTraits>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 23: launch_bulk_columns<IQ4XSTraits>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    case 42: launch_bulk_columns<Q20Traits>(weights, input, output, n_in, n_out, columns, width, tile_rows, st); break;
    }
    launch_check();
    const int first = columns / width * width;
    const int remainder = columns - first;
    for (int offset = first; offset < columns; offset += 8) {
        const auto* tail = static_cast<const uint8_t*>(input) +
            std::size_t(offset) * native_q8_1_bytes(n_in);
        native_mmvq(type, weights, tail, output + std::size_t(offset) * n_out,
            n_in, n_out, (columns - offset < 8 ? columns - offset : 8), stream);
    }
    return 1 + (remainder + 7) / 8;
}
