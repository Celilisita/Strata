#pragma once
#include "router_prediction.hpp"
struct ggml_backend_cuda_context;
namespace strata::deepseek_v4 {
bool router_prediction_reserve(ggml_backend_t);
void router_prediction_release(ggml_backend_cuda_context&);
void router_prediction_run(ggml_backend_cuda_context&,ggml_tensor*);
bool router_prefetch_probe_timing(ggml_backend_t,uint64_t*);
bool router_prediction_set_probe_epoch(ggml_backend_t,uint32_t);
bool router_prediction_probe_normalized(ggml_backend_t,float*);
uint32_t* native_tg_epoch_pointer(ggml_backend_cuda_context&);
}
