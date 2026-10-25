template<class F, bool Fp4, bool F16=false>
__global__ void encode_group_input(const Q81Block* source, unsigned* codes,
    GroupInputMeta* metadata, int width, int rows) {
    constexpr int BD=F16 ? 1 : Fp4 ? 3 : 2, Words=F::K/(F16 ? 2 : Fp4 ? 8 : 4);
    const int cell=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if(cell>=rows*(width/F::K)) return;
    const int group=cell/rows, token=cell%rows;
    const auto& x=source[size_t(token)*(width/32)+group/(32/F::K)];
    const int offset=(group%(32/F::K))*F::K;
    auto code=[&](int k) { return int(x.qs[offset+k]); };
    int sum=0;
#pragma unroll
    for(int k=0;k<F::K;++k) sum+=code(k);
    metadata[cell]={__half2float(__low2half(x.ds)),sum};
#pragma unroll
    for(int d=0;d<BD;++d)
#pragma unroll
        for(int w=0;w<Words;++w) codes[(size_t(cell)*BD+d)*Words+w]=group_code_word<Fp4,F16>(code,w,d,BD);
}
template<class F, bool Fp4, bool F16=false>
void launch_group_input(const void* input,void* packed,int width,int rows,void* stream) {
    auto* metadata=reinterpret_cast<GroupInputMeta*>(static_cast<unsigned char*>(packed)+size_t(rows)*width*2);
    const int cells=rows*(width/F::K);
    encode_group_input<F,Fp4,F16><<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(input),static_cast<unsigned*>(packed),metadata,width,rows);
}
