// Exact code/metadata transpose for coalesced canonical MMA loads.
// The temporary row-major producer remains canonical; no arithmetic changes.
template<class F, class Meta>
__global__ void native_tc_pack_transpose(const half* source, const Meta* metadata,
    half* destination, Meta* output_metadata, int n_in, int rows) {
    constexpr int R = 8, G = 32;
    const int tid = int(threadIdx.x), groups = n_in / 256 * F::T;
    const int row0 = int(blockIdx.x) * R, group0 = int(blockIdx.y) * G;
    const int row = tid / G, group = tid % G;
    const int actual_row = row0 + row, actual_group = group0 + group;
    __shared__ uint32_t codes[G][R * F::K + 1];
    __shared__ Meta meta[G][R + 1];
    const bool valid = actual_row < rows && actual_group < groups;
    if (valid) {
        const int kb = actual_group / F::T, iqs = F::kqs(actual_group % F::T);
        meta[group][row] = metadata[size_t(actual_row) * groups + actual_group];
#pragma unroll
        for (int part = 0; part < 2; ++part) {
#pragma unroll
            for (int k = 0; k < F::K; k += 2) {
                const size_t offset = size_t(actual_row) * n_in + kb * 256 + F::code_index(iqs, part, k);
                codes[group][row * F::K + part * (F::K / 2) + k / 2] =
                    reinterpret_cast<const uint32_t*>(source)[offset / 2];
            }
        }
    }
    __syncthreads();
    for (int item = tid; item < G * R * F::K; item += 256) {
        const int g = item / (R * F::K), r = item / F::K % R, word = item % F::K;
        if (row0 + r < rows && group0 + g < groups)
            reinterpret_cast<uint32_t*>(destination)[
                (size_t(group0 + g) * rows + row0 + r) * F::K + word] = codes[g][r * F::K + word];
    }
    const int g = tid / R, r = tid % R;
    if (row0 + r < rows && group0 + g < groups)
        output_metadata[size_t(group0 + g) * rows + row0 + r] = meta[g][r];
}
