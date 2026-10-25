#pragma once
#include <fp4_packed.hpp>
namespace strata::qwen35::native {
class DenseGemm;
uint64_t nvfp4_vector_unpack_submissions(DenseGemm&);
void fp4_unpack_vector(PackedFp4,void*,cudaStream_t);
}
