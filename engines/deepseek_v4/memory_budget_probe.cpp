#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-graph.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 3 && (argc != 5 || std::strcmp(argv[3], "--cache-mib"))) return 1;
    try {
        const uint64_t cache_mib = argc == 5 ? std::stoull(argv[4]) : 16384;
        if (cache_mib < 1 || cache_mib > 24576) throw std::runtime_error("metadata cache budget outside bound");
        llama_backend_init();
        auto* device = ggml_backend_dev_by_name("CUDA0");
        if (!device) throw std::runtime_error("CUDA0 unavailable");
        ggml_backend_dev_t devices[]{device, nullptr};
        const llama_model_tensor_buft_override overrides[]{
            {"blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr}};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.tensor_buft_overrides = overrides;
        mp.n_gpu_layers = -1; // Match runtime: dense output head also on GPU.
        mp.main_gpu = 0;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.use_extra_bufts = false;
        mp.load_mtp = false;
        mp.load_mode = LLAMA_LOAD_MODE_NONE; // no_alloc must not construct mmap-backed weight buffers.
        mp.lazy_mode = LLAMA_LAZY_MODE_ON;
        mp.no_alloc = true; // Metadata only; no model weight allocation/read.
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("metadata load failed");
        std::ofstream out(argv[2]);
        const char* stream_pp = std::getenv("STRATA_DSV4_STREAM_PP");
        out << "{\"weights_loaded\":false,\"inference_run\":false,\"ctx\":12288,\"batch\":8192,\"cache_mib\":" << cache_mib
            << ",\"native_stream_pp\":" << (stream_pp && !std::strcmp(stream_pp,"1") ? "true" : "false")
            << ",\"native_external_workspaces_excluded\":true,\"cases\":[";
        bool first = true;
        for (uint32_t ubatch : {4096u, 8192u}) {
            auto cp = llama_context_default_params();
            cp.n_ctx = 12288;
            cp.n_batch = 8192;
            cp.n_ubatch = ubatch;
            cp.n_seq_max = 1;
            cp.n_outputs_max = 1;
            cp.n_threads = cp.n_threads_batch = 4;
            cp.moe_cache_size = cache_mib << 20;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.type_k = cp.type_v = GGML_TYPE_F16;
            std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
            if (!context) throw std::runtime_error("metadata context reserve failed");
            if (!first) out << ',';
            first = false;
            out << "{\"ubatch\":" << ubatch << ",\"buffers\":[";
            bool first_buffer = true;
            uint64_t gpu_bytes = 0;
            for (const auto& entry : llama_get_memory_breakdown(context.get())) {
                const auto& costs = entry.second;
                const auto* name = ggml_backend_buft_name(entry.first);
                const bool gpu = std::strncmp(name, "CUDA", 4) == 0 && std::strcmp(name, "CUDA_Host") != 0;
                if (gpu) gpu_bytes += costs.model + costs.context + costs.compute;
                if (!first_buffer) out << ',';
                first_buffer = false;
                out << "{\"name\":\"" << name << "\",\"model\":" << costs.model
                    << ",\"context\":" << costs.context << ",\"compute\":" << costs.compute << '}';
            }
            out << "],\"sample_nodes\":[";
            bool first_node = true;
            auto* graph = context->get_gf_res_reserve()->get_gf();
            for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
                auto* tensor = ggml_graph_node(graph, index);
                const char* name = ggml_get_name(tensor);
                if (std::strncmp(name, "l_last-", 7) != 0 && std::strcmp(name, "h_nextn") != 0 &&
                    std::strncmp(name, "hc_head", 7) != 0 && std::strcmp(name, "result_norm") != 0 &&
                    std::strstr(name, "-37") == nullptr && std::strstr(name, "-38") == nullptr) continue;
                if (!first_node) out << ',';
                first_node = false;
                auto* backend = ggml_backend_sched_get_tensor_backend(context->get_sched(), tensor);
                out << "{\"name\":\"" << name << "\",\"type\":" << tensor->type << ",\"op\":\"" << ggml_op_name(tensor->op)
                    << "\",\"flags\":" << tensor->flags << ",\"backend\":\"" << (backend ? ggml_backend_name(backend) : "none") << "\",\"ne\":[";
                for (int dim = 0; dim < 4; ++dim) { if (dim) out << ','; out << tensor->ne[dim]; }
                out << "],\"nb\":[";
                for (int dim = 0; dim < 4; ++dim) { if (dim) out << ','; out << tensor->nb[dim]; }
                out << "],\"view\":" << (tensor->view_src ? "true" : "false") << '}';
            }
            out << "],\"canonical_gpu_bytes\":" << gpu_bytes
                << ",\"excludes_cuda_runtime_code_pool_and_display\":true}";
        }
        out << "]}\n";
        if (!out) throw std::runtime_error("report write failed");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
