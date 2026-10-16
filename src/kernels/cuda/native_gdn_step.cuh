// Canonical GDN recurrence; Columns only shares read-only Q/K/gate loads.
template<bool Scan, bool Prefix = false, int Columns = 1>
__global__ void __launch_bounds__(128, 2)
step(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
     const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
     float* __restrict__ output, int h_k, int h_v, float scale, int tokens, int input_stride,
     float* __restrict__ prefix) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * Columns;
    const int q_head = head % h_k;
    float s_shard[Columns][4], k_reg[4], q_reg[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
#pragma unroll
        for (int c = 0; c < Columns; ++c)
            s_shard[c][r] = state[(size_t(i) * h_v + head) * S + col + c];
    }
    const int count = Scan ? tokens : 1;
    for (int t = 0; t < count; ++t) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = r * 32 + lane;
            k_reg[r] = k[size_t(t) * input_stride + q_head * S + i];
            q_reg[r] = q[size_t(t) * input_stride + q_head * S + i];
        }
        const float g_val = expf(gate[size_t(t) * h_v + head]);
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) kv_shard += s_shard[c][r] * k_reg[r];
            const float kv_col = warp_sum(kv_shard);
            const float delta_col = (v[size_t(t) * input_stride + head * S + col + c] - g_val * kv_col) * beta[size_t(t) * h_v + head];
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                attn_partial += s_shard[c][r] * q_reg[r];
            }
            if (Prefix && t + 1 < count) {
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    prefix[size_t(t) * S * S * h_v +
                           (size_t(r * 32 + lane) * h_v + head) * S + col + c] = s_shard[c][r];
            }
            const float attn_col = warp_sum(attn_partial);
            if (lane == 0) output[(size_t(t) * h_v + head) * S + col + c] = attn_col * scale;
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
#pragma unroll
        for (int c = 0; c < Columns; ++c)
            state[(size_t(i) * h_v + head) * S + col + c] = s_shard[c][r];
    }
}

