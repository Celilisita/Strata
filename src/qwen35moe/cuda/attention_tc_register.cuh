// Register-resident online attention. Query/KV layout, causal masks, half
// probabilities and FP32 max/sum match the baseline; PV accumulations are
// reassociated across key tiles instead of spilled to shared after each tile.
// Included inside attention_tc.cu's anonymous namespace.
__global__ __launch_bounds__(128, 3)
void prefill_register(const float* q, const half* k, const half* v,
                      int first, int count, float* out) {
#if __CUDA_ARCH__ >= 700
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int head = blockIdx.x, kv = head / 8, qbase = blockIdx.y * Rows;
    const int active = min(Rows, count - qbase), width = first + qbase + active;
    __shared__ __align__(32) half query[Rows * Dim];
    union Phase {
        struct { float scores[Rows * Keys]; half probability[Rows * Keys]; } softmax;
        float output[Rows * Dim];
    };
    __shared__ __align__(32) Phase phase;
    __shared__ float stats[3 * Rows];
    auto* scores = phase.softmax.scores;
    auto* probability = phase.softmax.probability;
    wm::fragment<wm::accumulator, 16, 16, 16, float> accumulated[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wm::fill_fragment(accumulated[j], 0.0f);
    for (int i = tid; i < Rows * Dim; i += 128) {
        const int row = i / Dim, d = i % Dim;
        const float value = row < active ? q[(size_t(qbase + row) * Heads + head) * Dim + d] : 0;
        query[i] = __hmul(__float2half_rn(value * 0.0625f), __float2half_rn(0.25f));
    }
    if (tid < Rows) { stats[tid] = -CUDART_INF_F; stats[Rows + tid] = 0; }
    __syncthreads();
    for (int key = 0; key < width; key += Keys) {
        {
            wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> a;
            wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::col_major> b;
            wm::fragment<wm::accumulator, 16, 16, 16, float> product;
            wm::fill_fragment(product, 0.0f);
            for (int d = 0; d < Dim; d += 16) {
                wm::load_matrix_sync(a, query + d, Dim);
                wm::load_matrix_sync(b, k + size_t(key + warp * 16) * CacheStride + kv * Dim + d, CacheStride);
                wm::mma_sync(product, a, b, product);
            }
            wm::store_matrix_sync(scores + warp * 16, product, Keys, wm::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int own = 0; own < 4; ++own) {
            const int row = warp + own * 4;
            const bool valid = row < active;
            const int end = first + qbase + row + 1;
            const float s0 = valid && key + lane < end ? scores[row * Keys + lane] * 4.0f : -CUDART_INF_F;
            const float s1 = valid && key + lane + 32 < end ? scores[row * Keys + lane + 32] * 4.0f : -CUDART_INF_F;
            const float old_max = stats[row], old_sum = stats[Rows + row];
            const float m = valid ? fmaxf(old_max, maximum(fmaxf(s0, s1))) : 0;
            const float p0 = valid ? expf(s0 - m) : 0, p1 = valid ? expf(s1 - m) : 0;
            const float rescale = valid ? expf(old_max - m) : 0;
            const float denominator = valid ? old_sum * rescale + total(p0 + p1) : 1;
            probability[row * Keys + lane] = __float2half_rn(p0);
            probability[row * Keys + lane + 32] = __float2half_rn(p1);
            if (lane == 0) {
                stats[row] = m; stats[Rows + row] = denominator; stats[2 * Rows + row] = rescale;
            }
        }
        __syncthreads();
        // A WMMA load defines the accumulator element layout; do not assume an
        // undocumented per-lane row mapping when broadcasting the row scales.
        for (int i = tid; i < Rows * 16; i += 128) scores[i] = stats[2 * Rows + i / 16];
        __syncthreads();
        wm::fragment<wm::accumulator, 16, 16, 16, float> scale;
        wm::load_matrix_sync(scale, scores, 16, wm::mem_row_major);
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < accumulated[j].num_elements; ++i)
                accumulated[j].x[i] *= scale.x[i];
        wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> a;
        wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::row_major> b;
        for (int d = 0; d < Keys; d += 16) {
            wm::load_matrix_sync(a, probability + d, Keys);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wm::load_matrix_sync(b, v + size_t(key + d) * CacheStride + kv * Dim + warp * 64 + j * 16, CacheStride);
                wm::mma_sync(accumulated[j], a, b, accumulated[j]);
            }
        }
        // All probability reads must finish before the next scores overwrite.
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        wm::store_matrix_sync(phase.output + warp * 64 + j * 16, accumulated[j], Dim, wm::mem_row_major);
    __syncthreads();
    for (int i = tid; i < active * Dim; i += 128)
        out[(size_t(qbase + i / Dim) * Heads + head) * Dim + i % Dim] =
            phase.output[i] / stats[Rows + i / Dim];
#endif
}
