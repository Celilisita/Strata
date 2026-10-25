#include "pp_stream_graph.hpp"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <cstdlib>
#include <stdexcept>

namespace strata::deepseek_v4 {
static_assert(sizeof(ggml_map_custom3_op_params) == pp_stream_params_offset);
static void pp_stream_cpu_reject(ggml_tensor*, const ggml_tensor*, const ggml_tensor*,
                          const ggml_tensor*, int, int, void*) {
    GGML_ABORT("Strata MXFP4 PP operation requires the selected CUDA backend");
}
bool pp_stream_enabled(bool glm) {
    const char* v = std::getenv(glm ? "STRATA_GLM53_STREAM_PP" : "STRATA_DSV4_STREAM_PP");
    return v && v[0] == '1' && v[1] == '\0';
}
ggml_tensor* build_pp_stream(ggml_context* ctx, ggml_tensor* x, ggml_tensor* ids, ggml_tensor* weights,
                            ggml_tensor* gate, ggml_tensor* up, ggml_tensor* down, float clamp,
                            int window_experts, const PpCacheView* cache) {
    const char* window=std::getenv("STRATA_DSV4_PP_WINDOW_EXPERTS");
    if(window){char* end=nullptr;const long requested=std::strtol(window,&end,10);
        if(!*window||*end||(requested!=8&&requested!=16&&requested!=32))throw std::runtime_error("PP window must be8,16 or32 experts");window_experts=int(requested);}
    const bool d4 = gate && up && down && gate->ne[2] == 256 &&
        gate->type == GGML_TYPE_MXFP4 && up->type == gate->type && down->type == gate->type;
    const bool glm = gate && up && down && gate->ne[2] == 288 && gate->type == up->type &&
        ((gate->type == GGML_TYPE_IQ3_S && (down->type == GGML_TYPE_IQ4_XS || down->type == GGML_TYPE_Q6_K)) ||
         (gate->type == GGML_TYPE_IQ4_XS && down->type == GGML_TYPE_Q6_K));
    const int topk = glm ? 8 : 6;
    if ((!d4 && !glm) || !x || !ids || !weights || x->type != GGML_TYPE_F32 || x->ne[0] != 4096 ||
        x->ne[1] < 1 || x->ne[1] > (glm ? 16384 : 8192) || ids->type != GGML_TYPE_I32 || ids->ne[0] != topk ||
        ids->ne[1] != x->ne[1] || weights->type != GGML_TYPE_F32 ||
        ggml_nelements(weights) != x->ne[1] * topk || window_experts < 1 || window_experts > 64)
        throw std::runtime_error("Strata MXFP4 PP input geometry differs");
    for (auto* t : {gate, up, down}) {
        if (!t || t->ne[2] != (glm ? 288 : 256) || t->ne[3] != 1 ||
            !ggml_is_contiguous(t) || (t->data && (!t->buffer || !ggml_backend_buffer_is_host(t->buffer))))
            throw std::runtime_error("Strata MXFP4 PP requires immutable host model banks");
    }
    if (gate->ne[0] != 4096 || gate->ne[1] != 2048 || up->ne[0] != 4096 || up->ne[1] != 2048 ||
        down->ne[0] != 2048 || down->ne[1] != 4096)
        throw std::runtime_error("Strata MXFP4 expert geometry differs");
    // Pack the learned top-k view before the bounded host read.
    if (!ggml_is_contiguous(ids)) ids = ggml_cont(ctx, ids);
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    if (!ggml_is_contiguous(weights)) weights = ggml_cont(ctx, weights);
    auto* out = ggml_map_custom3(ctx, x, ids, weights, pp_stream_cpu_reject, 1,
                               reinterpret_cast<void*>(pp_stream_magic));
    const PpStreamSources s{gate, up, down, clamp, uint32_t(window_experts), cache};
    std::memcpy(reinterpret_cast<char*>(out->op_params) + pp_stream_params_offset, &s, sizeof(s));
    return out;
}
}
