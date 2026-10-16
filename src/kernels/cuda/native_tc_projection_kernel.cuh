// Benchmark candidate only. Production grouped kernel retains its accepted source.
template<class F>
__launch_bounds__(256, 3)
__global__ void native_tc_projection_kernel(const NativeTcTile* tiles,
    const Q81Block* inputs, float* output, int n_in, int n_out,
    const void* dense_weights = nullptr, int columns = 0, NativeTcPacked packed = {}) {
#if __CUDA_ARCH__ >= 700
    constexpr bool Dense = true;
    const NativeTcView<Dense> view(tiles, dense_weights, columns, packed);
    const auto* w = static_cast<const typename F::Block*>(view.weight());
    const int out_base = int(blockIdx.x) * 8, blocks = n_in / 256;
    const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    const int part = (lane >> 3) & 1, token_base = ((lane >> 2) & 1) * 8;
    __shared__ float lanes[32][129];
    for (int slice = 0; slice < 4; ++slice) {
        const int canonical_lane = warp * 4 + slice;
        float values[8] = {0,0,0,0,0,0,0,0};
        for (int native_warp = 0; native_warp < 4; ++native_warp) {
            const int native_tid = native_warp * 32 + canonical_lane;
            const int kb = native_tid / F::T;
            if (kb >= blocks) continue;
            if constexpr (Dense) {
                float partial[8] = {};
                for (int kbx = kb; kbx < blocks; kbx += F::BPI)
                    native_tc_product<F, true>(w, inputs, n_in, blocks, out_base,
                        native_tid, kbx, view, partial);
                if (!part) for (int j = 0; j < 8; ++j) {
                    if (!native_warp) values[j] = partial[j];
                    else values[j] = __fadd_rn(values[j], partial[j]);
                }
            } else {
                native_tc_product<F, false>(w, inputs, n_in, blocks, out_base,
                    native_tid, kb, view, values);
            }
        }
        if (!part) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = (lane & 1) + (j & 2) + (lane >= 16 ? 4 : 0);
                const int token = (j & 4) + (lane & 2) + (j & 1) + token_base;
                lanes[canonical_lane][row * 16 + token] = values[j];
            }
        }
    }
    __syncthreads();
    for (int cell = warp; cell < 128; cell += 8) {
        const int token = cell % 16, row = cell / 16;
        const float value = warp_sum(lanes[lane][cell]);
        if (lane == 0 && view.output_row(token) >= 0)
            output[size_t(view.output_row(token)) * n_out + out_base + row] = value;
    }
#endif
}
