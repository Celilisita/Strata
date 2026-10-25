struct alignas(16) BlackwellInputMeta { float d8[2]; int sums[2]; };
template<class F, bool Fp4, bool Int8 = false>
__global__ void encode_blackwell_input(const Q81Block* source, unsigned* codes,
    BlackwellInputMeta* metadata, int width, int rows) {
    constexpr int E = Fp4 ? 8 : 4, Bits = Fp4 ? 4 : 8, BD = Int8 ? 1 : Fp4 ? 3 : 2;
    constexpr int Words = 2*F::K/E;
    const int groups = (width/256)*F::T;
    const int cell = int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if (cell >= rows*groups) return;
    const int group = cell/rows, token = cell%rows, kb = group/F::T, iqs = F::kqs(group%F::T);
    typename F::W w{};
    if constexpr (F::K == 8) w.bq8_offset = 2*((iqs/2)/4);
    else w.bq8_offset = 4*(iqs/16)+(iqs%16)/8;
    const auto x = F::load_input(w,source+size_t(token)*(width/32)+kb*8,iqs);
    auto& m = metadata[cell];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        m.d8[p] = x.d8[p]; m.sums[p] = 0;
        if constexpr (F::K == 8)
#pragma unroll
            for (int k = 0; k < F::K; ++k) m.sums[p] += F::code(x,p,k);
    }
#pragma unroll
    for (int d = 0; d < BD; ++d)
#pragma unroll
        for (int word = 0; word < Words; ++word) {
            codes[(cell*BD+d)*Words+word] = blackwell_code_word<F,Fp4,false,Int8>(w,x,word,d);
        }
}
template<class F, bool Fp4, bool Int8 = false>
void launch_encode_input(const void* input, void* packed, int width, int rows, void* stream) {
    constexpr int BD = Int8 ? 1 : Fp4 ? 3 : 2;
    const size_t code_bytes = size_t(rows)*width*2;
    auto* meta = reinterpret_cast<BlackwellInputMeta*>(static_cast<unsigned char*>(packed)+code_bytes);
    const int cells = rows*(width/256)*F::T;
    encode_blackwell_input<F,Fp4,Int8><<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(input),static_cast<unsigned*>(packed),meta,width,rows);
}
