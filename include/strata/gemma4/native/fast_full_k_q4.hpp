#pragma once
#include "half_row_pair.hpp"
#include <cstdint>
namespace strata::gemma4::native {
struct FastFullKBinding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct FastFullKLease {HalfRowPairLease operands;FastFullKBinding binding;};
struct FastFullKResources {int registers,local_bytes,shared_bytes,max_threads;};
// Independent T1280 candidate. Continuous [Gate rows,Up rows] x5376 RN16
// weights; existing285/301 input scale and one FP16/FP32 GemmEx are unchanged.
void fast_full_k_q4_project(primitive::Matrix,primitive::Matrix,const FastFullKLease&,FastFullKBinding live);
FastFullKResources fast_full_k_q4_resources(); // cold only
namespace detail304 {
void validate(primitive::Matrix,primitive::Matrix,const FastFullKLease&,FastFullKBinding);
void weights(primitive::Matrix,primitive::Matrix,const HalfRowPairLease&,bool candidate);
}
}
