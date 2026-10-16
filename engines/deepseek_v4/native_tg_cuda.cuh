#pragma once
#include "native_tg_graph.hpp"
#include "native_tg_timing.hpp"
#include "ggml-backend.h"
struct ggml_backend_cuda_context;
namespace strata::deepseek_v4 {
void native_tg_run(ggml_backend_cuda_context&, ggml_tensor*);
bool native_tg_reserve(ggml_backend_t);
uintptr_t native_tg_slots(ggml_backend_t);
void native_tg_release(ggml_backend_cuda_context&);
uint32_t native_tg_ready(const DeviceMissPacket*);
void native_tg_reply(DeviceMissPacket*, uint32_t ticket);
void native_tg_complete(DeviceMissPacket*, uint32_t ticket, bool error);
}
