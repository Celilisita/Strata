#pragma once
#include "strata/qwen35/native/gdn.hpp"
#include <cuda_fp16.h>
#include <cstddef>
namespace strata::qwen35::native {
constexpr int GdnChunk=32,GdnKey=128;
struct GdnChunkWork {
    __half *w,*u,*b;float *prefix;
};
inline size_t gdn_chunk_workspace_bytes(GdnShape s){
    return size_t((s.tokens+31)/32)*s.value_heads*(2*32*128*2+32*32*2+32*4);
}
inline GdnChunkWork gdn_chunk_work(void *pointer,GdnShape s){
    const size_t blocks=size_t((s.tokens+31)/32)*s.value_heads;
    auto *half=static_cast<__half *>(pointer);auto *u=half+blocks*32*128,*b=u+blocks*32*128;
    return {half,u,b,reinterpret_cast<float *>(b+blocks*32*32)};
}
void gdn_chunk_prepare_configure();
void gdn_chunk_scan_configure();
void gdn_chunk_prepare(GdnInputs,GdnShape,GdnChunkWork,cudaStream_t);
void gdn_chunk_scan(GdnInputs,GdnShape,GdnChunkWork,float *,float *,cudaStream_t);
}
