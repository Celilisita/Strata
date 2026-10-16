// Exact integer codes, canonical quant metadata and Q8_1 scales/sums. Temporary
// views of caller storage; the authoritative weights/input remain unchanged.
template<int K> struct NativeTcInputMeta;
template<> struct NativeTcInputMeta<4> { uint32_t ds; };
template<> struct NativeTcInputMeta<8> { uint32_t ds; int sums[2]; };
struct NativeTcPacked {
    int weight_rows = 0, columns = 0;
    half* weight_codes = nullptr;
    NativeTcMeta* weight_meta = nullptr;
    half* input_codes = nullptr;
    void* input_meta = nullptr;
};

template<class F>
std::size_t native_tc_pack_bytes(int n_in, int n_out, int columns) {
    const size_t groups = size_t(n_in / 256) * F::T;
    return (size_t(n_out) + columns) * n_in * sizeof(half) +
        size_t(n_out) * groups * sizeof(NativeTcMeta) +
        size_t(columns) * groups * sizeof(NativeTcInputMeta<F::K>);
}

template<class F>
NativeTcPacked native_tc_pack_view(void* workspace, int n_in, int n_out, int columns) {
    auto* next = static_cast<uint8_t*>(workspace);
    const size_t groups = size_t(n_in / 256) * F::T;
    NativeTcPacked p;
    p.weight_rows = n_out; p.columns = columns;
    p.weight_codes = reinterpret_cast<half*>(next);
    next += size_t(n_in) * n_out * sizeof(half);
    p.weight_meta = reinterpret_cast<NativeTcMeta*>(next);
    next += size_t(n_out) * groups * sizeof(NativeTcMeta);
    p.input_codes = reinterpret_cast<half*>(next);
    next += size_t(n_in) * columns * sizeof(half);
    p.input_meta = next;
    return p;
}

template<class F>
__global__ void native_tc_pack_weights(const typename F::Block* weights,
    NativeTcPacked p, int n_in, int n_out) {
    const int groups = n_in / 256 * F::T;
    const int item = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (item >= n_out * groups) return;
    const int row = item / groups, kb = item / F::T % (n_in / 256);
    const int iqs = F::kqs(item % F::T);
    const auto a = F::load(weights + size_t(row) * (n_in / 256) + kb, iqs);
    p.weight_meta[item] = F::metadata(a);
#pragma unroll
    for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int k = 0; k < F::K; k += 2) {
            const size_t index = size_t(row) * n_in + kb * 256 + F::code_index(iqs, part, k);
            reinterpret_cast<uint32_t*>(p.weight_codes)[index / 2] =
                native_half_pair(F::weight_code(a, part, k), F::weight_code(a, part, k + 1));
        }
    }
}

template<class F>
__global__ void native_tc_pack_inputs(const Q81Block* inputs, NativeTcPacked p,
    int n_in, int columns) {
    const int groups = n_in / 256 * F::T;
    const int item = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (item >= columns * groups) return;
    const int row = item / groups, kb = item / F::T % (n_in / 256);
    const int iqs = F::kqs(item % F::T);
    const auto* x = inputs + size_t(row) * (n_in / 32) + kb * 8;
    NativeTcInputMeta<F::K> m{};
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int block = F::code_index(iqs, part, 0) / 32;
        m.ds |= uint32_t(__half_as_ushort(__low2half(x[block].ds))) << (16 * part);
#pragma unroll
        for (int k = 0; k < F::K; k += 2) {
            const int offset = F::code_index(iqs, part, k);
            const int a = x[offset / 32].qs[offset % 32];
            const int b = x[(offset + 1) / 32].qs[(offset + 1) % 32];
            reinterpret_cast<uint32_t*>(p.input_codes)[(size_t(row) * n_in + kb * 256 + offset) / 2] =
                native_half_pair(a, b);
            if constexpr (F::K == 8) m.sums[part] += a + b;
        }
    }
    static_cast<NativeTcInputMeta<F::K>*>(p.input_meta)[item] = m;
}
