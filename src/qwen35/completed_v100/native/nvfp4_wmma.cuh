#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace strata::completed_v100::qwen35::native {
void nvfp4_wmma_partial(bool tiled,const void* payload,const half* combined,float* partial,
    int rows,int columns,int tokens,int splits,cudaStream_t);
}
