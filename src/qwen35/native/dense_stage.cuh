#pragma once
// Two-stage decoded operands. All vector addresses are 16-byte aligned.
struct DenseFetch {uint4 weight,input0,input1;};
__device__ __forceinline__ uint4 dense_fetch4(const half *source,bool valid){
    uint4 out{};
    if(valid)asm volatile("ld.global.v4.u32 {%0,%1,%2,%3},[%4];"
                         : "=r"(out.x),"=r"(out.y),"=r"(out.z),"=r"(out.w)
                         : "l"(source) : "memory");
    return out;
}
__device__ __forceinline__ DenseFetch dense_fetch(const half *w,const half *x,int feature,int token,
                                                 int rows,int columns,int tokens,int base,int end){
    const int v=threadIdx.x,row=v/4,k=(v%4)*8;
    const bool av=feature+row<rows&&base+k<end,bv=token+row<tokens&&base+k<end,cv=token+row+64<tokens&&base+k<end;
    return {dense_fetch4(av?w+size_t(feature+row)*columns+base+k:w,av),
            dense_fetch4(bv?x+size_t(token+row)*columns+base+k:x,bv),
            dense_fetch4(cv?x+size_t(token+row+64)*columns+base+k:x,cv)};
}
__device__ __forceinline__ void dense_commit(DenseFetch value,half *stage){
    constexpr int Stride=40;
    const int row=threadIdx.x/4,k=(threadIdx.x%4)*8;
    *reinterpret_cast<uint4 *>(stage+row*Stride+k)=value.weight;
    *reinterpret_cast<uint4 *>(stage+(64+row)*Stride+k)=value.input0;
    *reinterpret_cast<uint4 *>(stage+(128+row)*Stride+k)=value.input1;
}
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void dense_copy_async(half *dest,const half *source,bool valid,const half *fallback){
    const unsigned address=unsigned(__cvta_generic_to_shared(dest));
    asm volatile("{ .reg .pred ignore; setp.eq.u32 ignore,%2,0; "
                 "cp.async.ca.shared.global [%0],[%1],16,ignore; }"
                 :: "r"(address),"l"(valid?source:fallback),"r"(int(valid)) : "memory");
}
__device__ __forceinline__ void dense_stage_async(const half *w,const half *x,half *stage,
                                                 int feature,int token,int rows,int columns,
                                                 int tokens,int base,int end){
    constexpr int Stride=40;
    const int row=threadIdx.x/4,k=(threadIdx.x%4)*8;
    const bool av=feature+row<rows&&base+k<end,bv=token+row<tokens&&base+k<end,cv=token+row+64<tokens&&base+k<end;
    dense_copy_async(stage+row*Stride+k,av?w+size_t(feature+row)*columns+base+k:w,av,w);
    dense_copy_async(stage+(64+row)*Stride+k,bv?x+size_t(token+row)*columns+base+k:x,bv,x);
    dense_copy_async(stage+(128+row)*Stride+k,cv?x+size_t(token+row+64)*columns+base+k:x,cv,x);
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void dense_wait_async(){asm volatile("cp.async.wait_group 0;" ::: "memory");}
#endif
