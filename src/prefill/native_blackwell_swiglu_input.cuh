// A canonical32-value Q8 block stays in one warp. Only its encoded consumer
// layout reaches global memory; optional canonical bytes belong to gates only.
template<bool Diagnostic>
__global__ void encode_swiglu_group_input(const float* __restrict__ gate,
    const float* __restrict__ up, unsigned* codes, GroupInputMeta* metadata,
    Q81Block* diagnostic, int width, int rows) {
    const int i = int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if (i >= rows*width) return; // width%32: every returning warp is complete.
    const int lane = int(threadIdx.x)%32, block = i/32;
    const float g = gate[i];
    const float x = __fmul_rn(__fdividef(g,__fadd_rn(1.0f,__expf(-g))),up[i]);
    float amax = fabsf(x), raw_sum = x;
#pragma unroll
    for (int offset=16;offset;offset>>=1) {
        amax = fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,offset,32));
        raw_sum += __shfl_xor_sync(0xffffffff,raw_sum,offset,32);
    }
    const float d = strata::kernels::q8_1_finite(amax/127.0f);
    const int q = int(strata::kernels::q8_1_quant(x,d,amax));
    int integer_sum = q;
#pragma unroll
    for (int offset=16;offset;offset>>=1)
        integer_sum += __shfl_xor_sync(0xffffffff,integer_sum,offset,32);
    if constexpr (Diagnostic) {
        diagnostic[block].qs[lane] = int8_t(q);
        if (!lane) diagnostic[block].ds = strata::kernels::q8_1_ds(d,raw_sum);
    }
    const int row = block/(width/32), group = block%(width/32);
    const size_t cell = size_t(group)*rows+row;
    if (!lane) metadata[cell] = {__half2float(__float2half(strata::kernels::q8_1_finite(d))),integer_sum};
    // All32 lanes execute each shuffle. Only then may8 word owners diverge.
    const int word = lane%8;
    const int q0 = __shfl_sync(0xffffffff,q,word*4,32);
    const int q1 = __shfl_sync(0xffffffff,q,word*4+1,32);
    const int q2 = __shfl_sync(0xffffffff,q,word*4+2,32);
    const int q3 = __shfl_sync(0xffffffff,q,word*4+3,32);
    if (lane<8) {
        const int values[4]{q0,q1,q2,q3};
        auto code = [&](int k) { return values[k-word*4]; };
#pragma unroll
        for (int digit=0;digit<2;++digit)
            codes[(cell*2+digit)*8+word] = group_code_word<false>(code,word,digit,2);
    }
}
