// Raw quantized integer codes: exact products, original F32 scale/finalizer.
__device__ __forceinline__ void blackwell_int8_mma(const unsigned* a,
    const unsigned* b, float* dots) {
    int d[4] = {};
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#pragma unroll
    for(int i=0;i<4;++i) dots[i] += float(d[i]);
}
