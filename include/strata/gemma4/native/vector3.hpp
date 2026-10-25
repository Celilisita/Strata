#pragma once
#include "strata/gemma4/native/ffn.hpp"
namespace strata::gemma4::native {
constexpr size_t ResidualVectorBlockBytes=120;
void prepare_vector3(const float*,void*,size_t,int,int,cudaStream_t);
void linear_vector3(primitive::Matrix,const void*,float*,int,cudaStream_t);
void pair_vector3(primitive::Matrix,primitive::Matrix,const void*,float*,float*,float*,int,cudaStream_t);
}
