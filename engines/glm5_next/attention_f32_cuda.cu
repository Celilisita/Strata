#include "common.cuh"
// This translation unit instantiates the existing F32 arithmetic tile path.
// All other CUDA translation units retain their original precision selection.
#undef FAST_FP16_AVAILABLE
#define STRATA_GLM53_F32_FLASH_TILE 1
#include "kv_window_attention_registry.cuh"
#include "attention_f32_load.cuh"
#include "fattn-tile.cuh"
#include "attention_f32.hpp"
#include <atomic>

namespace {
std::atomic<uint64_t> f32_dispatches{0},f16_dispatches{0},f16_heads32_dispatches{0};
}
namespace strata::glm5_next {
AttentionF32DispatchCounts attention_f32_dispatch_counts() {
    return {f32_dispatches.load(std::memory_order_relaxed),f16_dispatches.load(std::memory_order_relaxed),
        f16_heads32_dispatches.load(std::memory_order_relaxed)};
}
}

void strata_glm_attention_f32(ggml_backend_cuda_context& ctx,ggml_tensor* output) {
    GGML_ASSERT(strata::glm5_next::is_attention_f32(output));
    const auto* q=output->src[0];const auto* k=output->src[1];const auto* v=output->src[2];
    GGML_ASSERT(q->type==GGML_TYPE_F32&&((k->type==GGML_TYPE_F32&&v->type==GGML_TYPE_F32)||
        (strata::glm5_next::attention_f32_kv_direct_f16_eligible(q,k,v)||
         strata::glm5_next::attention_f32_kv_window_eligible(q,k,v))));
    GGML_ASSERT(q->ne[0]==512&&k->ne[0]==512&&v->ne[0]==512&&q->ne[2]==64&&k->ne[2]==1);
    const int heads_snapshot=output->op_params[strata::glm5_next::attention_f32_heads_index];
    GGML_ASSERT(heads_snapshot==0||heads_snapshot==32);
    GGML_ASSERT(heads_snapshot!=32||strata::glm5_next::attention_f32_kv_direct_f16_eligible(q,k,v));
    ggml_cuda_set_device(ctx.device);
    constexpr int queries=1,heads=16,columns=queries*heads;
    constexpr int configuration=ggml_cuda_fattn_tile_get_config_nvidia_fp32(512,512,columns);
    constexpr int threads=configuration&1023;
    constexpr int rows=(configuration>>14)&511;
    if(k->type==GGML_TYPE_Q8_0) {
        // True Q8 blocks are decoded directly into the existing F32 tile.
        // No full-history conversion or extra per-layer workspace is created.
        fattn_kernel_t quantized=flash_attn_tile<512,512,queries,heads,false,block_q8_0,true>;
        launch_fattn<512,queries,heads>(ctx,output,quantized,threads/32,0,rows,false,false,false,true,32);
        return;
    }
    if(k->type==GGML_TYPE_F16) {
        // T_KV changes source pointer units only; FAST_FP16_AVAILABLE remains
        // undefined, so shared tiles, Q/KQ/VKQ and all accumulators stay F32.
        f16_dispatches.fetch_add(1,std::memory_order_relaxed);
        // Cached device properties already belong to the original CUDA backend.
        // Unsupported devices retain16; only the selected32 branch counts as32.
        const auto& device=ggml_cuda_info().devices[ctx.device];
        if(heads_snapshot==32&&device.cc==GGML_CUDA_CC_BLACKWELL&&
           device.smpbo>=strata::glm5_next::attention_f32_heads32_shared_bytes) {
            constexpr int shared_bytes=int(strata::glm5_next::attention_f32_heads32_shared_bytes);
            constexpr int cfg=ggml_cuda_fattn_tile_get_config_nvidia_fp32(512,512,32);
            static_assert((cfg&1023)==256&&((cfg>>10)&15)==1&&((cfg>>14)&511)==32);
            fattn_kernel_t wide=flash_attn_tile<512,512,queries,32,false,half2,true,true>;
            // The existing macro configures this exact specialization once/device.
            // Its host check and first SetAttribute are paid eager/capture work.
            CUDA_SET_SHARED_MEMORY_LIMIT(wide,shared_bytes);
            f16_heads32_dispatches.fetch_add(1,std::memory_order_relaxed);
            launch_fattn<512,queries,32>(ctx,output,wide,256/32,shared_bytes,32,false,false,false,true,32);
            return;
        }
        fattn_kernel_t direct=flash_attn_tile<512,512,queries,heads,false,half2,true>;
        launch_fattn<512,queries,heads>(ctx,output,direct,threads/32,0,rows,false,false,false,true,32);
        return;
    }
    f32_dispatches.fetch_add(1,std::memory_order_relaxed);
    fattn_kernel_t kernel=flash_attn_tile<512,512,queries,heads,false,float2,true>;
    launch_fattn<512,queries,heads>(ctx,output,kernel,threads/32,0,rows,false,false,false,true,32);
}
