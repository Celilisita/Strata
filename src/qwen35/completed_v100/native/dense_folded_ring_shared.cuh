#pragma once
struct FoldedRingPrefix {uint4 weight,input;};
__device__ __forceinline__ uint4 folded_ring_read(const half *p){
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3},[%4];"
                 :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p));
    return v;
}
__device__ __forceinline__ const half *folded_ring_weight(const half *p,int v,int feature,int rows,int columns,int base){
    const int row=v/4,source=feature+row%32+(row/64)*32;
    const size_t plane=(row%64)>=32?size_t(rows)*columns:0;
    return p+plane+size_t(source)*columns+base+(v%4)*8;
}
__device__ __forceinline__ const half *folded_ring_input(const half *p,int v,int token,int columns,int base){
    return p+size_t(token+v/4)*columns+base+(v%4)*8;
}
__device__ __forceinline__ FoldedRingPrefix folded_ring_front(const half *w,const half *x,int tid,int feature,int token,int rows,int columns,int base){
    return {folded_ring_read(folded_ring_weight(w,tid,feature,rows,columns,base)),
            folded_ring_read(folded_ring_input(x,tid,token,columns,base))};
}
__device__ __forceinline__ void folded_ring_publish(half *shared,int stage,const FoldedRingPrefix &front,
        const half *w,const half *x,int tid,int feature,int token,int rows,int columns,int base){
    // Current MMA A/B registers are dead here: load the other two vectors now.
    const int v=tid+256;const uint4 wb=folded_ring_read(folded_ring_weight(w,v,feature,rows,columns,base));
    const uint4 xb=folded_ring_read(folded_ring_input(x,v,token,columns,base));
    half *a=shared+stage*256*40,*b=a+128*40;
    const int first=(tid/4)*40+(tid%4)*8,last=(v/4)*40+(v%4)*8;
    *reinterpret_cast<uint4 *>(a+first)=front.weight;*reinterpret_cast<uint4 *>(b+first)=front.input;
    *reinterpret_cast<uint4 *>(a+last)=wb;*reinterpret_cast<uint4 *>(b+last)=xb;
}
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void folded_ring_copy(half *dst,const half *src){
    const unsigned at=unsigned(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(at),"l"(src));
}
__device__ __forceinline__ void folded_ring_async(half *shared,int stage,const half *w,const half *x,
        int tid,int feature,int token,int rows,int columns,int base){
    half *a=shared+stage*256*40,*b=a+128*40;
    #pragma unroll
    for(int j=0;j<2;++j){
        const int v=tid+j*256,offset=(v/4)*40+(v%4)*8;
        folded_ring_copy(a+offset,folded_ring_weight(w,v,feature,rows,columns,base));
        folded_ring_copy(b+offset,folded_ring_input(x,v,token,columns,base));
    }
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void folded_ring_wait(){asm volatile("cp.async.wait_group 0;");}
#endif
