// Register matrix scheduling around the shared native quantized arithmetic.
// Each thread retains its original K chain; only row/token reuse changes.
struct PackedQ4 : Q4KTraits {
    __device__ static Input input(const W& w, const block_q8_1_mmq* x,
                                  int rows, int token, int kb, int iqs) {
        Input a;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int q = w.bq8_offset + p;
            const auto& b = x[size_t(kb*2+q/4)*rows+token];
            a.d8[p] = __low2float(b.ds4[q%4]);
            const auto* u = reinterpret_cast<const int*>(b.qs)+(q%4)*8+(iqs/2)%4;
            a.u[2*p] = u[0]; a.u[2*p+1] = u[4];
        }
        return a;
    }
};
struct PackedQ6 : Q6KTraits {
    __device__ static Input input(const W& w, const block_q8_1_mmq* x,
                                  int rows, int token, int kb, int iqs) {
        Input a;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int q = w.bq8_offset+2*p;
            const auto& b = x[size_t(kb*2+q/4)*rows+token];
            a.d8[p] = b.d4[q%4];
            a.u[p] = reinterpret_cast<const int*>(b.qs)[(q%4)*8+iqs%8];
        }
        return a;
    }
};
constexpr int CanonicalColumns = 8;
template<class F, bool Grouped, bool Active, int CanonicalRows = 2>
__launch_bounds__(128, 2)
__global__ void canonical_matrix_mmq(const typename F::Block* dense,
    const NativeMmqMatrix* matrices, const NativeMmqTile* tiles,
    const int32_t* output_ids, const block_q8_1_mmq* input, float* output,
    int width, int rows, int outputs, int stride, int parts) {
    const int tid = int(threadIdx.x), lane = tid%32, warp = tid/32;
    int begin = 0, end = rows, token0 = int(blockIdx.y)*CanonicalColumns;
    int part = 0;
    const typename F::Block* weights = dense;
    if constexpr (Grouped) {
        const auto tile = Active ? tiles[blockIdx.y/8] : NativeMmqTile{int(blockIdx.z)/parts, int(blockIdx.y)/8};
        const auto m = matrices[tile.matrix];
        begin = m.begin; end = m.end; part = int(blockIdx.z)%parts;
        token0 = begin + tile.tile*64 + int(blockIdx.y%8)*CanonicalColumns;
        weights = reinterpret_cast<const typename F::Block*>(uintptr_t(part ? m.up : m.gate));
    }
    if (token0 >= end) return;
    const int row0 = int(blockIdx.x)*CanonicalRows, blocks = width/256;
    const int iqs = F::kqs(tid);
    float sum[CanonicalColumns][CanonicalRows] = {};
    for (int kb = tid/F::T; kb < blocks; kb += F::BPI) {
        typename F::W w[CanonicalRows];
#pragma unroll
        for (int r = 0; r < CanonicalRows; ++r)
            w[r] = F::load(weights+size_t(row0+r)*blocks+kb,iqs);
#pragma unroll
        for (int c = 0; c < CanonicalColumns; ++c) {
            if (token0+c >= end) continue;
            const auto a = F::input(w[0],input,rows,token0+c,kb,iqs);
#pragma unroll
            for (int r = 0; r < CanonicalRows; ++r)
                sum[c][r] += F::apply_preloaded(w[r],a);
        }
    }
    __shared__ float partial[3][CanonicalColumns][CanonicalRows][32];
    if (warp) {
#pragma unroll
        for (int c = 0; c < CanonicalColumns; ++c)
#pragma unroll
            for (int r = 0; r < CanonicalRows; ++r)
                partial[warp-1][c][r][lane] = sum[c][r];
    }
    __syncthreads();
    if (warp) return;
#pragma unroll
    for (int c = 0; c < CanonicalColumns; ++c) {
        if (token0+c >= end) continue;
        const int dst = Grouped ? output_ids[token0+c] : token0+c;
#pragma unroll
        for (int r = 0; r < CanonicalRows; ++r) {
#pragma unroll
            for (int chain = 0; chain < 3; ++chain)
                sum[c][r] += partial[chain][c][r][lane];
#pragma unroll
            for (int offset = 16; offset; offset >>= 1)
                sum[c][r] += __shfl_xor_sync(0xffffffff,sum[c][r],offset);
            if (!lane) output[size_t(part)*rows*outputs+size_t(dst)*stride+row0+r] = sum[c][r];
        }
    }
}
