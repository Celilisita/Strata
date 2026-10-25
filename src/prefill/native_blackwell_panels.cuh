// CTA-local integer encoding. No model-weight mirror or global allocation.
template<class F, bool Fp4, bool Int8 = false>
__device__ void encode_blackwell_weights(const typename F::Block* weights, int weight_stride,
    int blocks, unsigned* aq) {
    constexpr int E = Fp4 ? 8 : 4, Bits = Fp4 ? 4 : 8;
    constexpr int AD = Int8 ? 1 : Fp4 ? F::WeightDigits4 : F::WeightDigits8, Words = 2*F::K/E;
    for (int cell = int(threadIdx.x); cell < blocks*F::T*8; cell += int(blockDim.x)) {
        const int group = cell/8, row = cell%8, kb = group/F::T, iqs = F::kqs(group%F::T);
        const auto* source = reinterpret_cast<const typename F::Block*>(
            reinterpret_cast<const unsigned char*>(weights)+row*weight_stride);
        const auto w = F::load(source+kb,iqs);
#pragma unroll
        for (int d = 0; d < AD; ++d)
#pragma unroll
            for (int word = 0; word < Words; ++word) {
                typename F::Input unused{};
                aq[(cell*AD+d)*Words+word] = blackwell_code_word<F,Fp4,true,Int8>(w,unused,word,d);
            }
    }
}
template<class F, bool Fp4, bool Int8 = false>
__device__ __forceinline__ void blackwell_encoded_product(const unsigned* aq, const unsigned* bq,
    int lane, float* dots) {
    constexpr int E = Fp4 ? 8 : 4, Bits = Fp4 ? 4 : 8, Words = 2*F::K/E;
    constexpr int AD = Int8 ? 1 : Fp4 ? F::WeightDigits4 : F::WeightDigits8, BD = Int8 ? 1 : Fp4 ? 3 : 2;
#pragma unroll
    for (int ad = 0; ad < AD; ++ad) {
        unsigned a[4] = {};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int start = (lane%4)*E;
            if (start < 2*F::K) {
                unsigned mask = 0xffffffff;
                if constexpr (F::K < E) mask = r ? 0xffff0000 : 0x0000ffff;
                else if (start/F::K != r) mask = 0;
                a[r] = aq[ad*Words+start/E] & mask;
            }
        }
#pragma unroll
        for (int bd = 0; bd < BD; ++bd) {
            unsigned b[2] = {};
            const int start = (lane%4)*E;
            if (start < 2*F::K) b[0] = bq[bd*Words+start/E];
            if constexpr (Int8) blackwell_int8_mma(a,b,dots);
            else blackwell_mma<Fp4>(a,b,dots,ad,bd);
        }
    }
}
template<class F, bool Fp4, bool Int8 = false> int blackwell_panel_bytes(int width) {
    constexpr int AD = Int8 ? 1 : Fp4 ? F::WeightDigits4 : F::WeightDigits8, BD = Int8 ? 1 : Fp4 ? 3 : 2;
    const int encoded = 8*width*(AD)*(Fp4 ? 4 : 8)/8;
    return 8*((width/256)*sizeof(typename F::Block)+4)+encoded;
}
