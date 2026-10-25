#pragma once
// Warp operands are loaded in wide transactions; all source rows align to16B.
__device__ __forceinline__ uint4 dense_shared4(const half *source){
    uint4 out;const unsigned address=unsigned(__cvta_generic_to_shared(source));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];"
                 : "=r"(out.x),"=r"(out.y),"=r"(out.z),"=r"(out.w) : "r"(address) : "memory");
    return out;
}
template<bool Upper>
__device__ __forceinline__ void dense_volta_step(uint4 a,uint4 b,float *c){
    const unsigned a0=Upper?a.z:a.x,a1=Upper?a.w:a.y;
    const unsigned b0=Upper?b.z:b.x,b1=Upper?b.w:b.y;
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),
                   "+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
                 : "r"(a0),"r"(a1),"r"(b0),"r"(b1));
}
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint4 dense_matrix4(const half *source){
    uint4 out;const unsigned address=unsigned(__cvta_generic_to_shared(source));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out.x),"=r"(out.y),"=r"(out.z),"=r"(out.w) : "r"(address) : "memory");
    return out;
}
__device__ __forceinline__ uint2 dense_matrix2(const half *source){
    uint2 out;const unsigned address=unsigned(__cvta_generic_to_shared(source));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out.x),"=r"(out.y) : "r"(address) : "memory");
    return out;
}
#endif
