#include "cpu_expert_service.hpp"
#include "canonical_expert_graph.hpp"

#include "ggml-cpu.h"
#include "llama-model.h"
#include <array>
#include <stdexcept>
#include <vector>

namespace strata::deepseek_v4 {
namespace {

struct Graph {
    llm_graph_result result{256};
    llama_adapter_loras loras;
    llm_graph_params params{};
    ggml_backend_sched_t scheduler = nullptr;
    std::unique_ptr<llm_graph_context> context;
    ggml_tensor *activation = nullptr, *ids = nullptr, *down = nullptr;

    Graph(const llama_model& model, ggml_backend_t cpu, int threads, int layer, uint32_t count) {
        ggml_backend_t backends[] = {cpu};
        scheduler = ggml_backend_sched_new(backends, nullptr, 1, 256, false, false);
        if (!scheduler) throw std::runtime_error("cold CPU expert scheduler allocation failed");
        try {
            params.arch = model.arch;
            params.hparams = model.hparams;
            params.hparams.n_expert_used_arr[layer] = count;
            params.cparams.n_ctx = 512;
            params.cparams.n_batch = params.cparams.n_ubatch = 1;
            params.cparams.n_threads = params.cparams.n_threads_batch = threads;
            params.cparams.op_offload = false;
            params.ubatch.n_tokens = 1;
            params.sched = scheduler;
            params.backend_cpu = cpu;
            params.loras = &loras;
            params.prec_policy = &model.prec_policy;
            params.n_outputs = 1;
            params.cb = [](const llama_ubatch&, ggml_tensor* t, const char* name, int il) {
                ggml_format_name(t, "%s-%d", name, il);
            };
            params.res = &result;
            context = std::make_unique<llm_graph_context>(params);
            activation = ggml_new_tensor_2d(context->ctx0, GGML_TYPE_F32, model.hparams.n_embd, 1);
            ids = ggml_new_tensor_2d(context->ctx0, GGML_TYPE_I32, count, 1);
            ggml_set_input(activation);
            ggml_set_input(ids);
            down = build_canonical_expert_down(*context, model, activation, ids, layer);
            if (!ggml_backend_sched_alloc_graph(scheduler, context->gf))
                throw std::runtime_error("cold CPU expert graph buffer allocation failed");
        } catch (...) {
            ggml_backend_sched_free(scheduler);
            scheduler = nullptr;
            throw;
        }
    }
    ~Graph() { if (scheduler) ggml_backend_sched_free(scheduler); }
};

void check_bank(const ggml_tensor* bank, int64_t input, int64_t output, int64_t count) {
    if (!bank || bank->type != GGML_TYPE_MXFP4 || bank->ne[0] != input || bank->ne[1] != output ||
        bank->ne[2] != count || bank->ne[3] != 1 || !ggml_is_contiguous(bank) || !bank->buffer ||
        !ggml_backend_buffer_is_host(bank->buffer))
        throw std::runtime_error("cold CPU expert service requires original contiguous host MXFP4 banks");
}

} // namespace

struct CpuExpertService::Impl {
    const llama_model& model;
    ggml_threadpool_t pool = nullptr;
    ggml_backend_t cpu = nullptr;
    std::vector<std::array<std::unique_ptr<Graph>, 6>> graphs;

    Impl(const llama_model& source, int threads) : model(source) {
        if (model.arch != LLM_ARCH_DEEPSEEK4 || model.hparams.n_embd != 4096 ||
            model.hparams.n_expert != 256 || model.layers.size() != 43 || threads < 1 || threads > 32)
            throw std::runtime_error("cold CPU expert service model or thread contract differs");
        try {
            auto options = ggml_threadpool_params_default(threads);
            pool = ggml_threadpool_new(&options);
            cpu = ggml_backend_cpu_init();
            if (!pool || !cpu) throw std::runtime_error("cold CPU expert resources unavailable");
            ggml_backend_cpu_set_n_threads(cpu, threads);
            ggml_backend_cpu_set_threadpool(cpu, pool);
            graphs.resize(model.layers.size());
            for (std::size_t il = 0; il < model.layers.size(); ++il) {
                const auto& layer = model.layers[il];
                if (model.hparams.n_expert_used(int(il)) != 6)
                    throw std::runtime_error("cold CPU expert service requires six original route ranks");
                check_bank(layer.ffn_gate_exps, 4096, 2048, 256);
                check_bank(layer.ffn_up_exps, 4096, 2048, 256);
                check_bank(layer.ffn_down_exps, 2048, 4096, 256);
                for (uint32_t count = 1; count <= 6; ++count)
                    graphs[il][count - 1] = std::make_unique<Graph>(model, cpu, threads, int(il), count);
            }
        } catch (...) { release(); throw; }
    }
    ~Impl() { release(); }
    void release() noexcept {
        graphs.clear();
        if (cpu) { ggml_backend_cpu_set_threadpool(cpu, nullptr); ggml_backend_free(cpu); cpu = nullptr; }
        if (pool) { ggml_threadpool_free(pool); pool = nullptr; }
    }
};

CpuExpertService::CpuExpertService(const llama_model& model, int threads) :
    impl_(std::make_unique<Impl>(model, threads)) {}
CpuExpertService::~CpuExpertService() = default;

void CpuExpertService::compute(int layer, const float* activation, const int32_t* ids,
                               std::size_t count, float* output) {
    if (layer < 0 || std::size_t(layer) >= impl_->graphs.size() || !activation || !ids || !output ||
        count < 1 || count > 6)
        throw std::runtime_error("cold CPU expert submission is invalid");
    for (std::size_t i = 0; i < count; ++i)
        if (ids[i] < 0 || ids[i] >= 256)
            throw std::runtime_error("cold CPU expert id is out of bounds");
    auto& graph = *impl_->graphs[layer][count - 1];
    ggml_backend_tensor_set(graph.activation, activation, 0, ggml_nbytes(graph.activation));
    ggml_backend_tensor_set(graph.ids, ids, 0, ggml_nbytes(graph.ids));
    const auto status = ggml_backend_sched_graph_compute(graph.scheduler, graph.context->gf);
    ggml_backend_sched_synchronize(graph.scheduler);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("canonical cold CPU expert compute failed");
    ggml_backend_tensor_get(graph.down, output, 0, ggml_nbytes(graph.down));
}

} // namespace strata::deepseek_v4
