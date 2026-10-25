#pragma once
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace strata::qwen35::native::nvfp4_detail {
__host__ __device__ inline size_t scale_index(int row,int group,int columns){
    const int groups4=(columns+63)/64;
    return size_t(row/128)*groups4*512+size_t(group/4)*512+
           (row%32)*16+((row%128)/32)*4+(group%4);
}
__device__ inline float scale_float(unsigned char b){
    return __half2float(__nv_cvt_fp8_to_halfraw(b,__NV_E4M3));
}
__device__ inline float fp4_float(unsigned char q){
    constexpr float values[8]={0,.5f,1,1.5f,2,3,4,6};
    return (q&8)?-values[q&7]:values[q&7];
}
__device__ inline float silu_product(float g,float u){return (g/(1.f+expf(-g)))*u;}
}
