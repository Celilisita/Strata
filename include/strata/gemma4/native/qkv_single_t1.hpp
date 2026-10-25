#pragma once
#include "strata/gemma4/native/vector_single_fast.hpp"
#include <cstdint>
namespace strata::gemma4::native {
struct QkvSingleBinding {
    const void* owner=nullptr;
    std::uint64_t epoch=0;
    cudaStream_t stream=nullptr;
};
struct QkvSingleLease {
    const float* normalized=nullptr;
    SingleQ8View input;
    QkvSingleBinding binding;
};
// Caller performs the unchanged RMS once. Producer is the existing196/243
// Single40 encoder; it is not assumed equal to179's first residual plane.
QkvSingleLease prepare_qkv_single_t1(const float*,void*,size_t,QkvSingleBinding);
// Explicit research adapter:244 Q4 striped or existing202→243 rawQ6 Single40.
// Head/output/FFN shapes and T>1 are excluded; no new CUDA or Model policy.
void project_qkv_single_t1(primitive::Matrix,QkvSingleLease,float*,QkvSingleBinding live);
}
