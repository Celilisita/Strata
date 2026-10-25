// Explicit FP16 PV experiment. QK, max and denominator remain FP32.
// Store a scaled numerator A=S*sum(exp(score-m)*V), with power-of-two S and
// S*denominator in (1/8,1/4]. Tile recurrence: A'=rescale*(S'/S)*A+S'*PV.
// Final A/(S*denominator) preserves real attention. Power-of-two changes avoid
// repeated normalization of a half accumulator. Quarter-V headroom does not
// clip V; the actual official-reference error gate owns numerical eligibility.
// Default dispatch stays FP32; independent official-reference error gate owns
// Ordinary runtime is bounded to visibleKV<=30720.
__device__ __forceinline__ float halfpv_numerator_scale(float denominator) {
    int exponent;
    const float fraction = frexpf(denominator, &exponent);
    return ldexpf(0.25f, (fraction == 0.5f ? 1 : 0) - exponent);
}
__global__ __launch_bounds__(128, 2)
void prefill_halfpv_query32(const float* q, const half* k, const half* v,
                     int first, int count, float* out) {
#if __CUDA_ARCH__ >= 700
    constexpr int QueryRows = 32;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int head = blockIdx.x, kv = head / 8, qbase = blockIdx.y * QueryRows;
    const int active = min(QueryRows, count - qbase), width = first + qbase + active;
    union Phase {
        struct {
            half query[QueryRows * Dim];
            float scores[QueryRows * Keys];
            half probability[QueryRows * Keys];
        } working;
        half output[QueryRows * Dim];
    };
    __shared__ __align__(32) Phase phase;
    __shared__ float stats[4 * QueryRows];
    auto* query = phase.working.query;
    auto* scores = phase.working.scores;
    auto* probability = phase.working.probability;
    wm::fragment<wm::accumulator, 16, 16, 16, half> accumulated[2][4];
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int j = 0; j < 4; ++j) wm::fill_fragment(accumulated[g][j], 0.0f);
    for (int i = tid; i < QueryRows * Dim; i += 128) {
        const int row = i / Dim, d = i % Dim;
        const float value = row < active ? q[(size_t(qbase + row) * Heads + head) * Dim + d] : 0;
        query[i] = __hmul(__float2half_rn(value * 0.0625f), __float2half_rn(0.25f));
    }
    if (tid < QueryRows) {
        stats[tid] = -CUDART_INF_F; stats[QueryRows + tid] = 0;
        stats[3 * QueryRows + tid] = 0;
    }
    __syncthreads();
    for (int key = 0; key < width; key += Keys) {
        {
            wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> a;
            wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::col_major> b;
            wm::fragment<wm::accumulator, 16, 16, 16, float> product[2];
#pragma unroll
            for (int g = 0; g < 2; ++g) wm::fill_fragment(product[g], 0.0f);
            for (int d = 0; d < Dim; d += 16) {
                wm::load_matrix_sync(b, k + size_t(key + warp * 16) * CacheStride + kv * Dim + d, CacheStride);
#pragma unroll
                for (int g = 0; g < 2; ++g) {
                    wm::load_matrix_sync(a, query + g * 16 * Dim + d, Dim);
                    wm::mma_sync(product[g], a, b, product[g]);
                }
            }
#pragma unroll
            for (int g = 0; g < 2; ++g)
                wm::store_matrix_sync(scores + g * 16 * Keys + warp * 16, product[g], Keys, wm::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int own = 0; own < 8; ++own) {
            const int row = warp + own * 4;
            const bool valid = row < active;
            const int end = first + qbase + row + 1;
            const float s0 = valid && key + lane < end ? scores[row * Keys + lane] * 4.0f : -CUDART_INF_F;
            const float s1 = valid && key + lane + 32 < end ? scores[row * Keys + lane + 32] * 4.0f : -CUDART_INF_F;
            const float old_max = stats[row], old_sum = stats[QueryRows + row];
            const float m = valid ? fmaxf(old_max, maximum(fmaxf(s0, s1))) : 0;
            const float p0 = valid ? expf(s0 - m) : 0, p1 = valid ? expf(s1 - m) : 0;
            const float rescale = valid ? expf(old_max - m) : 0;
            const float denominator = valid ? old_sum * rescale + total(p0 + p1) : 1;
            const float old_scale = stats[3 * QueryRows + row];
            const bool finite = !valid || (old_sum >= 0 && isfinite(m) &&
                isfinite(denominator) && denominator > 0 &&
                (old_sum == 0 || (old_scale > 0 && isfinite(old_scale))));
            const float scale = valid && finite ? halfpv_numerator_scale(denominator) : 0;
            const float factor = valid && finite && old_sum > 0 ? rescale * (scale / old_scale) : 0;
            probability[row * Keys + lane] = __float2half_rn(finite ? p0 * scale : 0);
            probability[row * Keys + lane + 32] = __float2half_rn(finite ? p1 * scale : 0);
            if (lane == 0) {
                stats[row] = m;
                stats[QueryRows + row] = finite ? denominator : -1;
                stats[2 * QueryRows + row] = factor;
                stats[3 * QueryRows + row] = scale;
            }
        }
        __syncthreads();
        auto* half_scale = reinterpret_cast<half*>(scores); // QK scores are dead.
        for (int i = tid; i < QueryRows * 16; i += 128)
            half_scale[i] = __float2half_rn(stats[2 * QueryRows + i / 16]);
        __syncthreads();
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            wm::fragment<wm::accumulator, 16, 16, 16, half> scale;
            wm::load_matrix_sync(scale, half_scale + g * 16 * 16, 16, wm::mem_row_major);
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < accumulated[g][j].num_elements; ++i)
                    accumulated[g][j].x[i] = __hmul(accumulated[g][j].x[i], scale.x[i]);
        }
        wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> a[2];
        wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::row_major> b;
        for (int d = 0; d < Keys; d += 16) {
#pragma unroll
            for (int g = 0; g < 2; ++g) wm::load_matrix_sync(a[g], probability + g * 16 * Keys + d, Keys);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wm::load_matrix_sync(b, v + size_t(key + d) * CacheStride + kv * Dim + warp * 64 + j * 16, CacheStride);
#pragma unroll
                for (int g = 0; g < 2; ++g) wm::mma_sync(accumulated[g][j], a[g], b, accumulated[g][j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            wm::store_matrix_sync(phase.output + g * 16 * Dim + warp * 64 + j * 16,
                                  accumulated[g][j], Dim, wm::mem_row_major);
    __syncthreads();
    for (int i = tid; i < active * Dim; i += 128)
        out[(size_t(qbase + i / Dim) * Heads + head) * Dim + i % Dim] =
            stats[QueryRows + i / Dim] > 0 ? __half2float(phase.output[i]) /
                (stats[QueryRows + i / Dim] * stats[3 * QueryRows + i / Dim]) : CUDART_NAN_F;
#endif
}
