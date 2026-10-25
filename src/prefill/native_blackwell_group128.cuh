// Two independent N64 subtiles share one lossless weight decode per K panel.
// Keep 256 threads and the original shared layout. Each output retains the
// original ascending quant-group chain and its own four FP32 accumulators.
template<class F, bool Active, bool Panel = false>
__launch_bounds__(256, 1)
__global__ void blackwell_group128_mmq(const NativeMmqMatrix* matrices,
    const NativeMmqTile* tiles, const int32_t* output_ids, float* output,
    int width, int rows, int outputs, int parts, NativeMmqEncodedInput encoded) {
    static_assert(F::K == 32 && F::AD8 == 1, "group128 supports Q4/Q5 FP8 only");
    constexpr int AD = F::AD8, Words = F::K / 4;
    constexpr int Stride = AD * Words + (AD * Words >= 8 ? 4 : 2);
    const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;
    const auto t = Active ? tiles[blockIdx.y] : NativeMmqTile{int(blockIdx.z) / parts, int(blockIdx.y)};
    const auto m = matrices[t.matrix];
    const int part = int(blockIdx.z) % parts, first = m.begin + t.tile * 128;
    if (first >= m.end) return; // Uniform CTA decision, before any barrier.
    const auto* weights = reinterpret_cast<const typename F::Block*>(uintptr_t(part ? m.up : m.gate));
    const int panel_width = Panel ? 512 : width, panel_blocks = panel_width / 256;
    const int blocks = width / 256, groups = panel_width / F::K, row0 = int(blockIdx.x) * 16;
    const int row_words = panel_blocks * int(sizeof(typename F::Block)) / 4, stride = (row_words + 1) * 4;
    const int raw_bytes = 16 * stride, meta_bytes = 16 * groups * int(sizeof(uint2));
    const int staging_bytes = raw_bytes > meta_bytes ? raw_bytes : meta_bytes;
    extern __shared__ unsigned char staging[];
    auto* aq = reinterpret_cast<unsigned*>(staging + staging_bytes);
    auto* wm = reinterpret_cast<uint2*>(staging);
    const int row = lane / 4, col = 2 * (lane % 4);
    int source_b[2], source0[2], source1[2];
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int token = first + warp * 8 + sub * 64;
        source_b[sub] = token + row < m.end ? __ldg(encoded.source_ids + token + row) : 0;
        source0[sub] = token + col < m.end ? __ldg(encoded.source_ids + token + col) : 0;
        source1[sub] = token + col + 1 < m.end ? __ldg(encoded.source_ids + token + col + 1) : 0;
    }
    const auto* bq = static_cast<const unsigned*>(encoded.codes);
    const auto* bm = static_cast<const GroupInputMeta*>(encoded.metadata);
    float total[2][4] = {};
    for (int panel = 0; panel < width / panel_width; ++panel) {
        for (int i = int(threadIdx.x); i < 16 * row_words; i += 256) {
            const auto* original = reinterpret_cast<const unsigned*>(
                weights + size_t(row0 + i / row_words) * blocks + panel * panel_blocks);
            reinterpret_cast<unsigned*>(staging)[(i / row_words) * (row_words + 1) + i % row_words] = original[i % row_words];
        }
        __syncthreads();
        uint2 saved[4];
        for (int i = int(threadIdx.x), j = 0; i < 16 * groups; i += 256, ++j) {
            const int g = i / 16, r = i % 16;
            const auto& w = reinterpret_cast<const typename F::Block*>(staging + r * stride)[g / (256 / F::K)];
            saved[j] = F::meta(w, g % (256 / F::K));
            auto code = [&](int k) { return F::code(w, g % (256 / F::K), k); };
#pragma unroll
            for (int word = 0; word < Words; ++word)
                aq[i * Stride + word] = group_code_word<false, false>(code, word, 0, AD);
        }
        __syncthreads();
        for (int i = int(threadIdx.x), j = 0; i < 16 * groups; i += 256, ++j) wm[i] = saved[j];
        __syncthreads();
      if (first + warp * 8 < m.end) {
#pragma unroll 1
       for (int g = 0; g < groups; ++g) {
        const int input_group = panel * groups + g;
        const auto w0 = wm[g * 16 + row], w1 = wm[g * 16 + row + 8];
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) if (first + warp * 8 + sub * 64 < m.end) {
            float dot[4] = {};
            group_product<F, false, false>(aq, bq, g, input_group, source_b[sub], encoded.source_rows, lane, dot);
            const auto x0 = bm[size_t(input_group) * encoded.source_rows + source0[sub]];
            const auto x1 = bm[size_t(input_group) * encoded.source_rows + source1[sub]];
            total[sub][0] = __fadd_rn(total[sub][0], F::finish(w0, x0.d8, x0.sum, dot[0]));
            total[sub][1] = __fadd_rn(total[sub][1], F::finish(w0, x1.d8, x1.sum, dot[1]));
            total[sub][2] = __fadd_rn(total[sub][2], F::finish(w1, x0.d8, x0.sum, dot[2]));
            total[sub][3] = __fadd_rn(total[sub][3], F::finish(w1, x1.d8, x1.sum, dot[3]));
        }
       }
      }
      if constexpr(Panel) __syncthreads(); // All warps, including both inactive tails.
    }
#pragma unroll
    for (int sub = 0; sub < 2; ++sub)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int token = first + warp * 8 + sub * 64;
                if (token + col + j < m.end) {
                    const int dst = output_ids[token + col + j];
                    output[size_t(part) * rows * outputs + size_t(dst) * outputs + row0 + row + h * 8] = total[sub][h * 2 + j];
                }
            }
}

template<class F> int group_fp8_shared_bytes(int width) {
    constexpr int Words = F::K / 4, Stride = F::AD8 * Words + (F::AD8 * Words >= 8 ? 4 : 2);
    const int raw = 16 * ((width / 256) * int(sizeof(typename F::Block)) + 4);
    const int meta = 16 * (width / F::K) * int(sizeof(uint2));
    return (raw > meta ? raw : meta) + 16 * (width / F::K) * Stride * 4;
}
template<class F, bool Active, bool Panel = false>
void launch_group128(const NativeMmqMatrix* matrices, int groups, int parts, const int32_t* ids,
    float* output, int rows, int max_rows, int width, int outputs, void* stream,
    const NativeMmqTile* tiles, int tile_count, NativeMmqEncodedInput encoded) {
    const int bytes = group_fp8_shared_bytes<F>(Panel ? 512 : width);
    if (bytes > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(blackwell_group128_mmq<F, Active, Panel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const dim3 grid(outputs / 16, Active ? tile_count : (max_rows + 127) / 128, Active ? parts : groups * parts);
    blackwell_group128_mmq<F, Active, Panel><<<grid, 256, bytes, static_cast<cudaStream_t>(stream)>>>(
        matrices, tiles, ids, output, width, rows, outputs, parts, encoded);
}
