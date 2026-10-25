#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace strata::qwen35::native {
// Scratch is the existing input and weight arena, borrowed serially on the
// provider stream. Original compressed matrices remain in their owned storage.
void original_fused_projection(bool q8,const void* payload,const float* scale_header,
    const float* input,float* output,int rows,int columns,int tokens,
    void* combined,size_t combined_half_values,float* partial,size_t partial_values,
    int sm_count,int tile,cudaStream_t stream);
// Explicit alternate layout entry; never pass tiled payload to a GGUF reader.
void original_fused_tiled_projection(const void* tiled_payload,const float* scale_header,
    const float* input,float* output,int rows,int columns,int tokens,
    void* combined,size_t combined_half_values,float* partial,size_t partial_values,
    int sm_count,int tile,cudaStream_t stream);
void nvfp4_tile_pack(const void* original,void* tiled,int rows,int columns,cudaStream_t);
void nvfp4_tile_unpack(const void* tiled,void* original,int rows,int columns,cudaStream_t);

}
