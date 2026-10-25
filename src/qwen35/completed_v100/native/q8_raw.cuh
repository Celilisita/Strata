#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
namespace strata::completed_v100::qwen35::native {
__device__ __forceinline__ float q8_raw_value(const void* data,int columns,int row,int column) {
    const auto* block=static_cast<const unsigned char*>(data)+(size_t(row)*(columns/32)+column/32)*34;
    const float delta=__half2float(*reinterpret_cast<const half*>(block));
    return delta*float(reinterpret_cast<const int8_t*>(block+2)[column%32]);
}
}
