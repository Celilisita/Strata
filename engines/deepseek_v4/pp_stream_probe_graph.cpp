#include "pp_stream_probe_graph.hpp"
#include "pp_stream_graph.hpp"
#include "ggml-alloc.h"
#include <algorithm>
#include <stdexcept>
namespace strata::deepseek_v4 {
PpProbeGraph::PpProbeGraph(ggml_backend_t b, const std::array<ggml_tensor*, 3>& source,
                          int tokens, float clamp, bool streamed, int window, const PpCacheView* cache) : backend(b) {
    meta = ggml_init({1ull << 20, nullptr, true});
    wmeta = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
    if (!meta || !wmeta) throw std::runtime_error("PP probe metadata allocation failed");
    x = ggml_new_tensor_2d(meta, GGML_TYPE_F32, 4096, tokens);
    ids_storage = ggml_new_tensor_2d(meta, GGML_TYPE_I32, streamed ? 256 : 6, tokens);
    ids = streamed ? ggml_view_2d(meta, ids_storage, 6, tokens, ids_storage->nb[1], 0) : ids_storage;
    weights = ggml_new_tensor_3d(meta, GGML_TYPE_F32, 1, 6, tokens);
    for (int i = 0; i < 3; ++i) {
        if (streamed) banks[i] = source[i];
        else banks[i] = ggml_new_tensor_3d(wmeta, GGML_TYPE_MXFP4,
                       source[i]->ne[0], source[i]->ne[1], 256);
    }
    if (streamed) {
        out = build_pp_stream(meta, x, ids, weights, banks[0], banks[1], banks[2], clamp, window, cache);
    } else {
        auto* act = ggml_reshape_3d(meta, x, 4096, 1, tokens);
        auto* gate = ggml_mul_mat_id(meta, banks[0], act, ids);
        auto* up = ggml_mul_mat_id(meta, banks[1], act, ids);
        auto* h = ggml_swiglu_clamp(meta, gate, up, clamp);
        auto* down = ggml_mul_mat_id(meta, banks[2], h, ids);
        auto* weighted = ggml_mul(meta, down, weights);
        out = ggml_view_2d(meta, weighted, 4096, tokens, weighted->nb[2], 0);
        for (int r = 1; r < 6; ++r)
            out = ggml_add(meta, out, ggml_view_2d(meta, weighted, 4096, tokens,
                                                 weighted->nb[2], r * weighted->nb[1]));
        out = ggml_cont(meta, out);
    }
    ggml_set_output(out);
    graph = ggml_new_graph_custom(meta, 128, false);
    ggml_build_forward_expand(graph, out);
    if (!streamed) {
        wbuffer = ggml_backend_alloc_ctx_tensors(wmeta, backend);
        if (!wbuffer) throw std::runtime_error("PP probe GPU weight allocation failed");
        ggml_backend_buffer_set_usage(wbuffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    buffer = ggml_backend_alloc_ctx_tensors(meta, backend);
    if (!buffer) throw std::runtime_error("PP probe GPU activation allocation failed");
}
PpProbeGraph::~PpProbeGraph() {
    ggml_backend_synchronize(backend);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (wbuffer) ggml_backend_buffer_free(wbuffer);
    if (meta) ggml_free(meta);
    if (wmeta) ggml_free(wmeta);
}
void PpProbeGraph::inputs(const std::vector<float>& xv, const std::vector<int32_t>& iv, const std::vector<float>& wv) {
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    if (ids_storage == ids) ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
    else {
        std::vector<int32_t> strided(size_t(ids_storage->ne[0]) * ids_storage->ne[1], -1);
        for (int t = 0; t < ids->ne[1]; ++t) std::copy_n(iv.data() + t * 6, 6, strided.data() + t * 256);
        ggml_backend_tensor_set(ids_storage, strided.data(), 0, strided.size() * sizeof(int32_t));
    }
    ggml_backend_tensor_set(weights, wv.data(), 0, wv.size() * sizeof(float));
}
void PpProbeGraph::upload(const std::array<ggml_tensor*, 3>& source) {
    for (int i = 0; i < 3; ++i)
        ggml_backend_tensor_set_async(backend, banks[i], source[i]->data, 0, ggml_nbytes(source[i]));
}
std::vector<float> PpProbeGraph::output() {
    std::vector<float> values(ggml_nelements(out));
    ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
    return values;
}
}
