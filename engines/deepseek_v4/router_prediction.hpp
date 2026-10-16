#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include <cstring>
#include <vector>
namespace strata::deepseek_v4 {
inline constexpr uint64_t router_prediction_tag=0x5354525052443031ull;
struct RouterPredictionParams {float norm_eps,hc_eps;int32_t layer;uint16_t gating,oracle;};
static_assert(sizeof(RouterPredictionParams)==16);
struct RouterPredictionRecord {
    int32_t top[4][10]{};
    float score[4][10]{};
    uint32_t missing[4]{};
    uint64_t start[4]{},end[4]{};
};
static_assert(sizeof(RouterPredictionRecord)==400);
struct RouterPredictionTrace {uint32_t epochs=0;std::vector<RouterPredictionRecord> records;};
inline bool is_router_prediction(const ggml_tensor* t) {
    if(!t||t->op!=GGML_OP_MAP_CUSTOM3)return false;
    uint64_t tag=0;std::memcpy(&tag,reinterpret_cast<const char*>(t->op_params)+16,8);return tag==router_prediction_tag;
}
inline RouterPredictionParams router_prediction_params(const ggml_tensor* t) {
    RouterPredictionParams p{};std::memcpy(&p,reinterpret_cast<const char*>(t->op_params)+24,sizeof(p));return p;
}
bool router_prediction_enabled();
ggml_tensor* build_router_prediction(ggml_context*,ggml_cgraph*,ggml_backend_sched_t,
    ggml_tensor* raw,ggml_tensor* merged,ggml_tensor* router,ggml_tensor* norm,
    ggml_tensor* hc_fn,ggml_tensor* hc_scale,ggml_tensor* hc_base,ggml_tensor* bias,
    ggml_tensor* directory,ggml_backend_t backend,const RouterPredictionParams&);
bool router_prediction_trace_read(ggml_backend_t,RouterPredictionTrace*,bool);
}
