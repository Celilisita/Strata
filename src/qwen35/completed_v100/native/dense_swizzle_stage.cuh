#pragma once
#include "dense_stage.cuh"
// Permute four K8 vectors per row, preserving each 16-byte payload. The
// row-bit mix serves both Volta quad-pairs and SM80+ eight-row ldmatrix.
__device__ __forceinline__ int dense_swizzle(int row){
    return (((row>>1)&1)|((((row>>2)^(row>>3))&1)<<1))*8;
}
__device__ __forceinline__ int dense_swizzle_index(int row,int k){
    return row*32+(k^dense_swizzle(row));
}
__device__ __forceinline__ uint4 dense_swizzle_load4(unsigned address){
    uint4 out;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];"
                 : "=r"(out.x),"=r"(out.y),"=r"(out.z),"=r"(out.w) : "r"(address) : "memory");
    return out;
}
__device__ __forceinline__ void dense_swizzle_commit(DenseFetch value,half *stage){
    const int row=threadIdx.x/4,k=(threadIdx.x%4)*8;
    const int offset=dense_swizzle_index(row,k);
    *reinterpret_cast<uint4 *>(stage+offset)=value.weight;
    *reinterpret_cast<uint4 *>(stage+64*32+offset)=value.input0;
    *reinterpret_cast<uint4 *>(stage+128*32+offset)=value.input1;
}
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void dense_swizzle_async(const half *w,const half *x,half *stage,
                                                   int feature,int token,int rows,int columns,
                                                   int tokens,int base,int end){
    const int row=threadIdx.x/4,k=(threadIdx.x%4)*8;
    const int offset=dense_swizzle_index(row,k);
    const bool av=feature+row<rows&&base+k<end,bv=token+row<tokens&&base+k<end,cv=token+row+64<tokens&&base+k<end;
    dense_copy_async(stage+offset,av?w+size_t(feature+row)*columns+base+k:w,av,w);
    dense_copy_async(stage+64*32+offset,bv?x+size_t(token+row)*columns+base+k:x,bv,x);
    dense_copy_async(stage+128*32+offset,cv?x+size_t(token+row+64)*columns+base+k:x,cv,x);
    asm volatile("cp.async.commit_group;" ::: "memory");
}
#endif
