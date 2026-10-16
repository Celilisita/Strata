// Volta PTX fragment layout: NVIDIA PTX ISA, mma.m8n8k4 f16 section.
// Four independent8x8 products per warp. Register operands contain exact
// integer codes; bounded4/8-term F32 sums remain exact integers.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m8n8k4-with-f16-floating-point-type
__device__ __forceinline__ uint32_t native_half_pair(int a, int b) {
    const half2 pair = __halves2half2(__int2half_rn(a), __int2half_rn(b));
    return *reinterpret_cast<const uint32_t*>(&pair);
}
__device__ __forceinline__ void native_mma_884(uint32_t a0, uint32_t a1,
    uint32_t b0, uint32_t b1, float* c) {
#if __CUDA_ARCH__ >= 700
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}
