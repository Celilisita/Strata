#pragma once
#include "ggml-backend.h"
#include <array>
#include <cstdint>
#include <vector>
namespace strata::deepseek_v4 {
struct PpCacheView;
struct PpProbeGraph {
    ggml_backend_t backend;
    ggml_context *meta = nullptr, *wmeta = nullptr;
    ggml_backend_buffer_t buffer = nullptr, wbuffer = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *x = nullptr, *ids = nullptr, *weights = nullptr, *out = nullptr;
    ggml_tensor* ids_storage = nullptr;
    std::array<ggml_tensor*, 3> banks{};
    PpProbeGraph(ggml_backend_t, const std::array<ggml_tensor*, 3>& sources,
                 int tokens, float clamp, bool streamed, int window, const PpCacheView* cache = nullptr);
    ~PpProbeGraph();
    void inputs(const std::vector<float>&, const std::vector<int32_t>&, const std::vector<float>&);
    void upload(const std::array<ggml_tensor*, 3>&);
    std::vector<float> output();
};
}
