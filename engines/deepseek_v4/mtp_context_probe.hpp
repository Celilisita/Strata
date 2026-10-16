#pragma once
#include "llama-context.h"
#include "llama-model.h"
#include "device_cache.hpp"
#include "ggml-cpu.h"
#include <memory>
#include <vector>
namespace strata::deepseek_v4 {
inline void mtp_context_probe(llama_model& model,ggml_backend_t backend,ggml_tensor* embedding) {
    // Reserve the authentic main graph and initialize its cache under pressure.
    // Dense tensors have borrowed dummy storage and no inference is performed.
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> dummy(ggml_backend_alloc_buffer(backend,6ull<<30),ggml_backend_buffer_free);
    if(!dummy)throw std::runtime_error("MTP context fixture dummy storage unavailable");ggml_backend_buffer_set_usage(dummy.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for(auto entry:model.tensors_by_name){auto* t=entry.second;if(!t->data){if(ggml_nbytes(t)>ggml_backend_buffer_get_size(dummy.get()))throw std::runtime_error("MTP metadata tensor exceeds dummy storage");t->data=ggml_backend_buffer_get_base(dummy.get());t->buffer=dummy.get();}}
    model.tok_embd->data=embedding->data;model.tok_embd->buffer=embedding->buffer;
    auto cp=llama_context_default_params();cp.n_ctx=12288;cp.n_batch=cp.n_ubatch=8192;cp.n_seq_max=1;cp.n_outputs_max=2;cp.n_rs_seq=2;
    cp.n_threads=cp.n_threads_batch=4;cp.moe_cache_size=16384ull<<20;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(&model,cp),llama_free);if(!context)throw std::runtime_error("MTP authentic graph context reserve failed");
    llama_synchronize(context.get());configure_device_expert_cache(context.get());llama_synchronize(context.get());
    context->strata_capture_nextn_tail(128);llama_memory_clear(llama_get_memory(context.get()),true);llama_synchronize(context.get());
    verify_device_expert_cache(context.get());context.reset();
    for(auto entry:model.tensors_by_name){auto* t=entry.second;if(t->buffer==dummy.get()){t->data=nullptr;t->buffer=nullptr;}}
    model.tok_embd->data=nullptr;model.tok_embd->buffer=nullptr;
}
}
