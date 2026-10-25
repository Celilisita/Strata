#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace strata::qwen35::native {
// A reversible nibble permutation: same36 bytes/block, all scales unchanged.
// Called only at a drained GPU inventory boundary, never during generation.
void nvfp4_repack_gpu(void* payload,size_t blocks,bool inverse,cudaStream_t);
}
