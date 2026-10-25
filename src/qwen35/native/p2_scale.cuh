#pragma once
#include <cuda_runtime.h>
namespace strata::qwen35::native::p2_scale {
__device__ __forceinline__ int power(float maximum){
    const unsigned bits=__float_as_uint(maximum);
    return max(-14,min(9,int((bits>>23)&255)-127-6+int((bits&0x7fffff)>0x7e0000)));
}
__device__ __forceinline__ float scale(int p){return __uint_as_float(unsigned(p+127)<<23);}
__device__ __forceinline__ int code(float value,int p){
    return max(-127,min(127,__float2int_rn(__fmul_rn(value,__uint_as_float(unsigned(127-p)<<23)))));
}
}
