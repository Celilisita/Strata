#include "cpu_expert_service.hpp"
#include "canonical_expert_graph.hpp"
#include "cpu_partial_down_graph.hpp"
#include "../glm5_next/cpu_placement.hpp"
#include "../glm5_next/cpu_derived_layout.hpp"
#include "../glm5_next/cpu_iq3_r4_layout.hpp"
#include "../glm5_next/cpu_iq3_exact_lut_diagnostic.hpp"
#include "../glm5_next/cpu_gu_q8_reuse.hpp"
#include "../glm5_next/cpu_gu_q8_policy.hpp"

#include "ggml-cpu.h"
#include "llama-model.h"
#include <array>
#include <cstring>
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
    std::unique_ptr<strata::glm5_next::CpuGuQ8Reuse> gu_q8;
    std::unique_ptr<CpuPartialDownGraph> partial_down;

    Graph(const llama_model& model, ggml_backend_t cpu, int threads, int layer, uint32_t count,
          bool routed_rows=false,bool gate_up_only=false,bool gu_q8_reuse=false,size_t partial_down_routes=0) {
        ggml_backend_t backends[] = {cpu};
        scheduler = ggml_backend_sched_new(backends, nullptr, 1, 256, false, false);
        if (!scheduler) throw std::runtime_error("cold CPU expert scheduler allocation failed");
        try {
            params.arch = model.arch;
            params.hparams = model.hparams;
            const uint32_t tokens=routed_rows?count:1;
            const uint32_t used=routed_rows?1:count;
            params.hparams.n_expert_used_arr[layer] = used;
            params.cparams.n_ctx = 512;
            params.cparams.n_batch = params.cparams.n_ubatch = tokens;
            params.cparams.n_threads = params.cparams.n_threads_batch = threads;
            params.cparams.op_offload = false;
            params.ubatch.n_tokens = tokens;
            params.sched = scheduler;
            params.backend_cpu = cpu;
            params.loras = &loras;
            params.prec_policy = &model.prec_policy;
            params.n_outputs = tokens;
            params.cb = [](const llama_ubatch&, ggml_tensor* t, const char* name, int il) {
                ggml_format_name(t, "%s-%d", name, il);
            };
            params.res = &result;
            context = std::make_unique<llm_graph_context>(params);
            activation = ggml_new_tensor_2d(context->ctx0, GGML_TYPE_F32, model.hparams.n_embd, tokens);
            ids = ggml_new_tensor_2d(context->ctx0, GGML_TYPE_I32, used, tokens);
            ggml_set_input(activation);
            ggml_set_input(ids);
            down = build_canonical_expert_down(*context, model, activation, ids, layer);
            if(gu_q8_reuse) {
                if(!routed_rows||gate_up_only||model.arch!=LLM_ARCH_GLM5_NEXT)
                    throw std::runtime_error("GU Q8 reuse requires original routed full FFN");
                const auto& row=model.layers.at(layer);
                gu_q8=std::make_unique<strata::glm5_next::CpuGuQ8Reuse>(context->ctx0,context->gf,
                    activation,ids,down,std::array<ggml_tensor*,2>{row.ffn_gate_exps,row.ffn_up_exps},scheduler,cpu,threads);
            }
            if (gate_up_only) {
                auto* hidden = down->src[1];
                if (!routed_rows || down->op != GGML_OP_MUL_MAT_ID ||
                    down->src[0] != model.layers.at(layer).ffn_down_exps || !hidden ||
                    hidden->type != GGML_TYPE_F32 || hidden->op != GGML_OP_GLU ||
                    (ggml_get_glu_op(hidden) != GGML_GLU_OP_SWIGLU &&
                     ggml_get_glu_op(hidden) != GGML_GLU_OP_SWIGLU_CLAMP) ||
                    hidden->ne[0] != 2048 || hidden->ne[1] != 1 ||
                    hidden->ne[2] != count || hidden->ne[3] != 1 || !ggml_is_contiguous(hidden))
                    throw std::runtime_error("diagnostic canonical Gate/Up boundary differs");
                // Retain the real canonical SwiGLU ancestor, without executing Down.
                down = hidden;
                ggml_graph_clear(context->gf);
                ggml_set_output(down);
                ggml_build_forward_expand(context->gf, down);
            }
            if (partial_down_routes) {
                const auto& row = model.layers.at(layer);
                if (!routed_rows || gate_up_only || gu_q8_reuse || model.arch != LLM_ARCH_GLM5_NEXT ||
                    threads != 16 || layer != 24 || row.ffn_gate_exps->type != GGML_TYPE_IQ3_S ||
                    row.ffn_up_exps->type != GGML_TYPE_IQ3_S)
                    throw std::runtime_error("partial Down requires original L24 IQ3 GU and native16");
                partial_down = std::make_unique<CpuPartialDownGraph>(context->ctx0, context->gf,
                    down, ids, row.ffn_down_exps, count, partial_down_routes);
                down = partial_down->output;
                for (int i = 0; i < ggml_graph_n_nodes(context->gf); ++i)
                    if (!ggml_backend_supports_op(cpu, ggml_graph_node(context->gf, i)))
                        throw std::runtime_error("partial Down graph node is unsupported by the existing CPU backend");
            }
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

void check_bank(const ggml_tensor* bank, int64_t input, int64_t output, int64_t count, bool glm) {
    if (!bank || (glm ? (bank->type != GGML_TYPE_IQ3_S && bank->type != GGML_TYPE_IQ4_XS && bank->type != GGML_TYPE_Q6_K && bank->type != GGML_TYPE_Q3_K && bank->type != GGML_TYPE_Q4_K) : bank->type != GGML_TYPE_MXFP4) || bank->ne[0] != input || bank->ne[1] != output ||
        bank->ne[2] != count || bank->ne[3] != 1 || !ggml_is_contiguous(bank) || !bank->buffer ||
        !ggml_backend_buffer_is_host(bank->buffer))
        throw std::runtime_error("CPU expert service requires original contiguous host quantized banks");
}

} // namespace

struct CpuExpertService::Impl {
    const llama_model& model;
    ggml_threadpool_t pool = nullptr;
    ggml_backend_t cpu = nullptr;
    std::vector<std::vector<std::unique_ptr<Graph>>> graphs;
    int threads = 0;
    std::vector<std::vector<std::unique_ptr<Graph>>> row_graphs;
    std::vector<float> row_activation;
    std::vector<std::vector<std::unique_ptr<Graph>>> gu_q8_graphs;
    bool diagnostic_gu_q8=false;
    bool runtime_gu_q8_requested=false,runtime_gu_q8_active=false;
    std::vector<std::vector<std::unique_ptr<Graph>>> gate_up_graphs;
    std::vector<float> gate_up_activation;
    std::unique_ptr<Graph> partial_down_graph;
    const strata::glm5_next::CpuDerivedLayout* diagnostic_derived = nullptr;
    const strata::glm5_next::CpuIq3R4Layout* diagnostic_iq3_r4 = nullptr;
    const strata::glm5_next::CpuIq3LutDiagnostic* diagnostic_exact_lut = nullptr;
    bool routed_rows_prepared = false;
    bool four_routed_rows_prepared = false;

    void prepare_row_shape(int layer,size_t routes) {
        if(row_graphs.size()<graphs.size())row_graphs.resize(graphs.size());
        if(row_activation.size()<routes*4096)row_activation.resize(routes*4096);
        auto& variants=row_graphs[layer];
        if(variants.size()<routes)variants.resize(routes);
        if(!variants[routes-1])
            variants[routes-1]=std::make_unique<Graph>(model,cpu,threads,layer,uint32_t(routes),true);
    }

    Impl(const llama_model& source, int thread_count) : model(source), threads(thread_count) {
        const bool glm = model.arch == LLM_ARCH_GLM5_NEXT;
        runtime_gu_q8_requested=glm&&strata::glm5_next::requested_cpu_gu_q8_reuse();
        runtime_gu_q8_active=runtime_gu_q8_requested&&strata::glm5_next::requested_cpu_gu_q8_active();
        if ((!glm && model.arch != LLM_ARCH_DEEPSEEK4) || model.hparams.n_embd != 4096 ||
            model.hparams.n_expert != (glm ? 288 : 256) || (!glm && model.layers.size() != 43) || threads < 1 || threads > 32)
            throw std::runtime_error("cold CPU expert service model or thread contract differs");
        try {
            auto options = ggml_threadpool_params_default(threads);
            if (glm) strata::glm5_next::configure_cpu_pool(options);
            const auto creator_affinity=glm?strata::glm5_next::save_cpu_affinity():CpuCoreBinding{};
            pool = ggml_threadpool_new(&options);
            if(glm)strata::glm5_next::restore_cpu_affinity(creator_affinity);
            cpu = ggml_backend_cpu_init();
            if (!pool || !cpu) throw std::runtime_error("cold CPU expert resources unavailable");
            ggml_backend_cpu_set_n_threads(cpu, threads);
            ggml_backend_cpu_set_threadpool(cpu, pool);
            graphs.resize(model.layers.size());
            for (std::size_t il = 0; il < model.layers.size(); ++il) {
                const auto& layer = model.layers[il];
                if (!layer.ffn_up_exps) continue;
                const int used = int(model.hparams.n_expert_used(int(il)));
                if (used != (glm ? 8 : 6)) throw std::runtime_error("cold CPU route count differs");
                check_bank(layer.ffn_gate_exps, 4096, 2048, model.hparams.n_expert, glm);
                check_bank(layer.ffn_up_exps, 4096, 2048, model.hparams.n_expert, glm);
                check_bank(layer.ffn_down_exps, 2048, 4096, model.hparams.n_expert, glm);
                graphs[il].resize(used);
                if(!glm)for (uint32_t count = 1; count <= uint32_t(used); ++count)
                    graphs[il][count - 1] = std::make_unique<Graph>(model, cpu, threads, int(il), count);
            }
        } catch (...) { release(); throw; }
    }
    ~Impl() { release(); }
    void release() noexcept {
        if (pool && diagnostic_derived) strata_glm_cpu_derived_bind(pool,nullptr);
        diagnostic_derived = nullptr;
        if (pool && diagnostic_iq3_r4) strata_glm_cpu_iq3_r4_bind(pool,nullptr);
        diagnostic_iq3_r4 = nullptr;
        if (pool && diagnostic_exact_lut) strata_glm_cpu_iq3_lut_bind(pool,nullptr);
        diagnostic_exact_lut = nullptr;
        partial_down_graph.reset();
        gu_q8_graphs.clear();
        gate_up_graphs.clear();
        row_graphs.clear();
        graphs.clear();
        if (cpu) { ggml_backend_cpu_set_threadpool(cpu, nullptr); ggml_backend_free(cpu); cpu = nullptr; }
        if (pool) { ggml_threadpool_free(pool); pool = nullptr; }
    }
};

CpuExpertService::CpuExpertService(const llama_model& model, int threads) :
    impl_(std::make_unique<Impl>(model, threads)) {}
CpuExpertService::~CpuExpertService() = default;

bool CpuExpertService::set_diagnostic_derived(const strata::glm5_next::CpuDerivedLayout* layout) noexcept {
    if (!impl_ || !impl_->pool) return false;
    if (layout) {
        if (impl_->diagnostic_gu_q8 || impl_->diagnostic_derived || impl_->diagnostic_iq3_r4 || impl_->diagnostic_exact_lut || impl_->model.arch != LLM_ARCH_GLM5_NEXT || impl_->threads != 16) return false;
        bool original_pair = false;
        for (const auto& layer : impl_->model.layers)
            if (layer.ffn_gate_exps == layout->gate() && layer.ffn_up_exps == layout->up()) original_pair = true;
        if (!original_pair) return false;
    }
    if (!strata_glm_cpu_derived_bind(impl_->pool,layout ? layout->view() : nullptr)) return false;
    impl_->diagnostic_derived = layout;
    return true;
}

bool CpuExpertService::set_diagnostic_iq3_r4(const strata::glm5_next::CpuIq3R4Layout* layout) noexcept {
    if (!impl_ || !impl_->pool) return false;
    if (layout) {
        if (impl_->diagnostic_gu_q8 || impl_->diagnostic_iq3_r4 || impl_->diagnostic_derived || impl_->diagnostic_exact_lut ||
            impl_->model.arch != LLM_ARCH_GLM5_NEXT || impl_->threads != 16) return false;
        bool original_pair = false;
        for (const auto& layer : impl_->model.layers)
            if (layer.ffn_gate_exps == layout->gate() && layer.ffn_up_exps == layout->up()) original_pair = true;
        if (!original_pair) return false;
    }
    if (!strata_glm_cpu_iq3_r4_bind(impl_->pool,layout ? layout->view() : nullptr)) return false;
    impl_->diagnostic_iq3_r4 = layout;
    return true;
}

bool CpuExpertService::set_diagnostic_exact_lut(const strata::glm5_next::CpuIq3LutDiagnostic* diagnostic) noexcept {
    if (!impl_ || !impl_->pool) return false;
    if (diagnostic) {
        if (impl_->diagnostic_gu_q8 || impl_->diagnostic_exact_lut || impl_->diagnostic_iq3_r4 || impl_->diagnostic_derived ||
            impl_->model.arch != LLM_ARCH_GLM5_NEXT || impl_->threads != 16) return false;
        bool original_pair = false;
        for (const auto& layer : impl_->model.layers)
            if (layer.ffn_gate_exps == diagnostic->gate() && layer.ffn_up_exps == diagnostic->up()) original_pair = true;
        if (!original_pair) return false;
    }
    if (!strata_glm_cpu_iq3_lut_bind(impl_->pool,diagnostic ? diagnostic->view() : nullptr)) return false;
    impl_->diagnostic_exact_lut = diagnostic;
    return true;
}

void CpuExpertService::prepare() {
    for(size_t layer=0;layer<impl_->graphs.size();++layer)
        for(size_t count=1;count<=impl_->graphs[layer].size();++count){
            auto& graph=impl_->graphs[layer][count-1];
            if(!graph)graph=std::make_unique<Graph>(impl_->model,impl_->cpu,impl_->threads,int(layer),uint32_t(count));
        }
}
void CpuExpertService::compute(int layer, const float* activation, const int32_t* ids,
                               std::size_t count, float* output) {
    if (layer < 0 || std::size_t(layer) >= impl_->graphs.size() || !activation || !ids || !output ||
        count < 1 || count > impl_->graphs[layer].size())
        throw std::runtime_error("cold CPU expert submission is invalid");
    for (std::size_t i = 0; i < count; ++i)
        if (ids[i] < 0 || ids[i] >= int32_t(impl_->model.hparams.n_expert))
            throw std::runtime_error("cold CPU expert id is out of bounds");
    auto& owner = impl_->graphs[layer][count - 1];
    if(!owner)owner = std::make_unique<Graph>(impl_->model, impl_->cpu, impl_->threads, layer, uint32_t(count));
    auto& graph = *owner;
    ggml_backend_tensor_set(graph.activation, activation, 0, ggml_nbytes(graph.activation));
    ggml_backend_tensor_set(graph.ids, ids, 0, ggml_nbytes(graph.ids));
    ggml_status status=GGML_STATUS_FAILED;
    auto compute=[&]{status=ggml_backend_sched_graph_compute(graph.scheduler,graph.context->gf);ggml_backend_sched_synchronize(graph.scheduler);};
    if(impl_->model.arch==LLM_ARCH_GLM5_NEXT)strata::glm5_next::with_cpu_caller(impl_->threads,compute);else compute();
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("canonical cold CPU expert compute failed");
    ggml_backend_tensor_get(graph.down, output, 0, ggml_nbytes(graph.down));
}

void CpuExpertService::prepare_routed_rows() {
    if(impl_->model.arch!=LLM_ARCH_GLM5_NEXT)
        throw std::runtime_error("routed-row CPU service is only validated for GLM");
    if(impl_->row_activation.size()<16*4096)impl_->row_activation.resize(16*4096);
    impl_->row_graphs.resize(impl_->graphs.size());
    for(size_t layer=0;layer<impl_->graphs.size();++layer) {
        if(impl_->graphs[layer].empty())continue;
        auto& variants=impl_->row_graphs[layer];if(variants.size()<16)variants.resize(16);
        const auto& banks=impl_->model.layers[layer];
        const bool reuse=impl_->runtime_gu_q8_requested&&strata::glm5_next::eligible_cpu_gu_q8_reuse(
            banks.ffn_gate_exps,banks.ffn_up_exps,impl_->threads);
        for(size_t routes=1;routes<=16;++routes){impl_->prepare_row_shape(int(layer),routes);
            if(reuse)prepare_gu_q8_reuse_diagnostic(int(layer),routes);}
    }
    impl_->routed_rows_prepared=true;
}

void CpuExpertService::compute_routed_rows(int layer,const float* activations,const int32_t* ids,
    const uint8_t* positions,size_t routes,float* output) {
    // Diagnostic preparation must not opt the production entry into wider shapes.
    if(!impl_->routed_rows_prepared||routes<1||routes>16)
        throw std::runtime_error("two-position routed-row CPU service was not prepared before the miss");
    compute_routed_rows_impl(layer,activations,2,ids,positions,routes,output);
}

void CpuExpertService::prepare_routed_rows_four() {
    if(impl_->model.arch!=LLM_ARCH_GLM5_NEXT||impl_->threads!=16)
        throw std::runtime_error("four-position CPU service requires original GLM native16");
    if(impl_->row_activation.size()<32*4096)impl_->row_activation.resize(32*4096);
    for(size_t layer=0;layer<impl_->graphs.size();++layer) {
        if(impl_->graphs[layer].empty())continue;
        for(size_t routes=1;routes<=32;++routes)impl_->prepare_row_shape(int(layer),routes);
    }
    impl_->four_routed_rows_prepared=true;
}

void CpuExpertService::compute_routed_rows_four(int layer,const float* activations,const int32_t* ids,
    const uint8_t* positions,size_t routes,float* output) {
    if(!impl_->four_routed_rows_prepared||impl_->model.arch!=LLM_ARCH_GLM5_NEXT||
       impl_->threads!=16||routes<1||routes>32)
        throw std::runtime_error("four-position routed-row CPU service was not prepared before the miss");
    compute_routed_rows_impl(layer,activations,4,ids,positions,routes,output);
}

CpuExpertService::RoutedRowsCapacity CpuExpertService::routed_rows_capacity() const {
    RoutedRowsCapacity out;
    out.activation_capacity_bytes=impl_->row_activation.capacity()*sizeof(float);
    for(const auto* variants:{&impl_->row_graphs,&impl_->gu_q8_graphs})
        for(const auto& layer:*variants){bool typed=false;
            for(const auto& graph:layer)if(graph){
                ++out.prepared_graphs;
                if(graph->gu_q8){++out.gu_q8_prepared_graphs;typed=true;}
                out.scheduler_cpu_buffer_bytes+=ggml_backend_sched_get_buffer_size(graph->scheduler,impl_->cpu);
            }
            if(typed)++out.gu_q8_prepared_layers;
        }
    if (impl_->partial_down_graph) {
        ++out.prepared_graphs;
        out.scheduler_cpu_buffer_bytes += ggml_backend_sched_get_buffer_size(
            impl_->partial_down_graph->scheduler, impl_->cpu);
    }
    return out;
}

void CpuExpertService::prepare_routed_rows_diagnostic(int layer,size_t routes) {
    if(impl_->model.arch!=LLM_ARCH_GLM5_NEXT||impl_->threads!=16||layer<0||
       size_t(layer)>=impl_->graphs.size()||impl_->graphs[layer].empty()||routes<1||routes>64)
        throw std::runtime_error("diagnostic routed-row CPU shape requires GLM CPU16 and1..64 real routes");
    impl_->prepare_row_shape(layer,routes);
}

void CpuExpertService::compute_routed_rows_diagnostic(int layer,const float* activations,size_t positions_count,
    const int32_t* ids,const uint8_t* positions,size_t routes,float* output) {
    if(impl_->model.arch!=LLM_ARCH_GLM5_NEXT||impl_->threads!=16||positions_count<1||positions_count>8)
        throw std::runtime_error("diagnostic routed-row CPU service requires GLM CPU16 and1..8 activation rows");
    compute_routed_rows_impl(layer,activations,positions_count,ids,positions,routes,output);
}

void CpuExpertService::prepare_gate_up_diagnostic(int layer,size_t routes) {
    if(impl_->model.arch!=LLM_ARCH_GLM5_NEXT||impl_->threads!=16||layer<0||
       size_t(layer)>=impl_->graphs.size()||impl_->graphs[layer].empty()||routes<1||routes>16)
        throw std::runtime_error("diagnostic Gate/Up shape requires GLM CPU16 and1..16 actual routes");
    if(impl_->gate_up_graphs.size()<impl_->graphs.size())impl_->gate_up_graphs.resize(impl_->graphs.size());
    if(impl_->gate_up_activation.size()<routes*4096)impl_->gate_up_activation.resize(routes*4096);
    auto& variants=impl_->gate_up_graphs[layer];
    if(variants.size()<routes)variants.resize(routes);
    if(!variants[routes-1])
        variants[routes-1]=std::make_unique<Graph>(impl_->model,impl_->cpu,impl_->threads,layer,
            uint32_t(routes),true,true);
}

void CpuExpertService::compute_gate_up_diagnostic(int layer,const float* activations,const int32_t* ids,
    const uint8_t* positions,size_t routes,float* output) {
    if(impl_->model.arch!=LLM_ARCH_GLM5_NEXT||impl_->threads!=16||layer<0||
       size_t(layer)>=impl_->gate_up_graphs.size()||!activations||!ids||!positions||!output||
       routes<1||routes>16||impl_->gate_up_graphs[layer].size()<routes||
       !impl_->gate_up_graphs[layer][routes-1]||impl_->gate_up_activation.size()<routes*4096)
        throw std::runtime_error("diagnostic Gate/Up graph was not prepared before compute");
    for(size_t row=0;row<routes;++row) {
        if(positions[row]>1||ids[row]<0||ids[row]>=int32_t(impl_->model.hparams.n_expert))
            throw std::runtime_error("diagnostic Gate/Up key or activation position differs");
        std::memcpy(impl_->gate_up_activation.data()+row*4096,
            activations+size_t(positions[row])*4096,4096*sizeof(float));
    }
    auto& graph=*impl_->gate_up_graphs[layer][routes-1];
    ggml_backend_tensor_set(graph.activation,impl_->gate_up_activation.data(),0,ggml_nbytes(graph.activation));
    ggml_backend_tensor_set(graph.ids,ids,0,ggml_nbytes(graph.ids));
    ggml_status status=GGML_STATUS_FAILED;
    strata::glm5_next::with_cpu_caller(impl_->threads,[&]{
        status=ggml_backend_sched_graph_compute(graph.scheduler,graph.context->gf);
        ggml_backend_sched_synchronize(graph.scheduler);
    });
    if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("canonical diagnostic Gate/Up compute failed");
    ggml_backend_tensor_get(graph.down,output,0,ggml_nbytes(graph.down));
}

void CpuExpertService::prepare_gu_q8_reuse_diagnostic(int layer,size_t routes) {
    if(impl_->model.arch!=LLM_ARCH_GLM5_NEXT||impl_->threads!=16||layer<0||
        size_t(layer)>=impl_->graphs.size()||impl_->graphs[layer].empty()||routes<1||routes>16)
        throw std::runtime_error("GU Q8 reuse preparation requires original GLM native16 and1..16 routes");
    impl_->prepare_row_shape(layer,routes);
    if(impl_->gu_q8_graphs.size()<impl_->graphs.size())impl_->gu_q8_graphs.resize(impl_->graphs.size());
    auto& variants=impl_->gu_q8_graphs[layer];if(variants.size()<routes)variants.resize(routes);
    if(!variants[routes-1])variants[routes-1]=std::make_unique<Graph>(impl_->model,impl_->cpu,
        impl_->threads,layer,uint32_t(routes),true,false,true);
}
bool CpuExpertService::gu_q8_reuse_runtime_requested()const noexcept {
    return impl_&&impl_->runtime_gu_q8_requested;
}
bool CpuExpertService::gu_q8_reuse_runtime_active()const noexcept {
    return impl_&&impl_->runtime_gu_q8_active;
}
bool CpuExpertService::set_gu_q8_reuse_runtime_active(bool enabled) noexcept {
    if(!impl_||(enabled&&!impl_->runtime_gu_q8_requested))return false;
    impl_->runtime_gu_q8_active=enabled;return true;
}
bool CpuExpertService::set_diagnostic_gu_q8_reuse(bool enabled) noexcept {
    if(!impl_||!impl_->pool)return false;
    if(enabled&&(impl_->model.arch!=LLM_ARCH_GLM5_NEXT||impl_->threads!=16||
        impl_->diagnostic_derived||impl_->diagnostic_iq3_r4||impl_->diagnostic_exact_lut))return false;
    impl_->diagnostic_gu_q8=enabled;return true;
}
CpuExpertService::GuQ8ReuseSnapshot CpuExpertService::gu_q8_reuse_snapshot(int layer,size_t routes)const {
    if(layer<0||size_t(layer)>=impl_->gu_q8_graphs.size()||routes<1||
        impl_->gu_q8_graphs[layer].size()<routes||!impl_->gu_q8_graphs[layer][routes-1])
        throw std::runtime_error("GU Q8 reuse snapshot shape was not prepared");
    const auto& input=*impl_->gu_q8_graphs[layer][routes-1]->gu_q8;const auto count=input.counts();
    GuQ8ReuseSnapshot out;out.routes=input.routes();out.input_bytes=input.input_bytes();
    out.packed_bytes=input.packed_bytes();out.worker_calls=count.calls;
    out.quantized_blocks=count.quantized_blocks;out.copied_blocks=count.copied_blocks;out.packed=input.snapshot();return out;
}

void CpuExpertService::compute_routed_rows_impl(int layer,const float* activations,size_t positions_count,
    const int32_t* ids,const uint8_t* positions,size_t routes,float* output) {
    if(layer<0||size_t(layer)>=impl_->row_graphs.size()||!activations||!ids||!positions||!output||
       routes<1||routes>64||impl_->row_graphs[layer].size()<routes||!impl_->row_graphs[layer][routes-1]||
       impl_->row_activation.size()<routes*4096)
        throw std::runtime_error("routed-row CPU service was not prepared before the miss");
    Graph* selected=impl_->row_graphs[layer][routes-1].get();
    if(impl_->diagnostic_gu_q8||(impl_->runtime_gu_q8_active&&impl_->routed_rows_prepared&&positions_count==2)) {
        const bool prepared=positions_count==2&&routes<=16&&size_t(layer)<impl_->gu_q8_graphs.size()&&
            impl_->gu_q8_graphs[layer].size()>=routes&&impl_->gu_q8_graphs[layer][routes-1];
        if(prepared)selected=impl_->gu_q8_graphs[layer][routes-1].get();
        else if(impl_->diagnostic_gu_q8)
            throw std::runtime_error("GU Q8 reuse exact B2 graph was not prepared before compute");
        // Unsupported GU formats/shapes retain the already prepared stock graph.
    }
    for(size_t row=0;row<routes;++row) {
        if(size_t(positions[row])>=positions_count||ids[row]<0||ids[row]>=int32_t(impl_->model.hparams.n_expert))
            throw std::runtime_error("routed-row CPU key or activation position differs");
        // One row for each actual missing route occurrence. No padded ranks,
        // union computation, second expert math, or per-expert executor.
        if(!selected->gu_q8)std::memcpy(impl_->row_activation.data()+row*4096,
            activations+size_t(positions[row])*4096,4096*sizeof(float));
    }
    auto& graph=*selected;
    if(graph.gu_q8)graph.gu_q8->set_inputs(activations,positions);
    else ggml_backend_tensor_set(graph.activation,impl_->row_activation.data(),0,ggml_nbytes(graph.activation));
    ggml_backend_tensor_set(graph.ids,ids,0,ggml_nbytes(graph.ids));
    ggml_status status=GGML_STATUS_FAILED;
    strata::glm5_next::with_cpu_caller(impl_->threads,[&]{status=ggml_backend_sched_graph_compute(graph.scheduler,graph.context->gf);ggml_backend_sched_synchronize(graph.scheduler);});
    if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("canonical routed-row CPU expert compute failed");
    ggml_backend_tensor_get(graph.down,output,0,ggml_nbytes(graph.down));
}

#include "cpu_partial_down_service.inc"

} // namespace strata::deepseek_v4
