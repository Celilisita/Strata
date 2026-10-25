#pragma once
// Eight-half vectors retain their order; row-dependent XOR permutes vectors.
// Rows separated by16 have the same permutation on both matrix-load bodies.
template<bool Swizzle,bool Gray=false>
__device__ __forceinline__ int joint_offset(int row,int k){
    static_assert(!(Swizzle&&Gray),"shared layouts are exclusive");
    if constexpr(Gray){
        const int shift=((row^(row>>1))&7)*8;
        return ((row<<6)|shift)^k;
    }else if constexpr(Swizzle){
        const int shift=((row&3)|((((row>>2)^(row>>3))&1)<<2))*8;
        return ((row<<6)|shift)^k;
    }else return row*72+k;
}
