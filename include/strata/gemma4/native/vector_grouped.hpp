#pragma once
#include "strata/gemma4/native/vector3.hpp"
namespace strata::gemma4::native {
void linear_vector_grouped(primitive::Matrix,const void*,float*,int,cudaStream_t);
void pair_vector_grouped(primitive::Matrix,primitive::Matrix,const void*,float*,float*,float*,int,cudaStream_t);
}
