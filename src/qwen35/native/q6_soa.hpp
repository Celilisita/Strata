#pragma once
#include "strata/qwen35/native/ffn.hpp"
namespace strata::qwen35::native {
// Private diagnostic layout: three natural-order I8 planes, then one float4
// (the existing rounded-F16 scales widened exactly) per 32-value group.
size_t q6_soa_bytes(int columns);
void q6_soa_prepare(const float*,void*,int columns,int tokens,cudaStream_t);
void q6_soa_project(Matrix,const void*,float*,int tokens,cudaStream_t);
void q6_soa_compare_encoding(const void*,const void*,int columns,int tokens,unsigned*,cudaStream_t);
}
