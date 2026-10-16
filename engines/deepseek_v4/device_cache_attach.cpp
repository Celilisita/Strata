#include "device_cache.hpp"
#include "phase_allocator.hpp"
#include "native_tg_graph.hpp"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-moe-cache.h"
#include "llama-model.h"
#include <stdexcept>
#include <cstdlib>
#include <cstring>

void llama_context::strata_set_device_cache() {
    if (!moe_cache || strata_device_cache || strata_cold || strata_lease)
        throw std::runtime_error("device cache requires one canonical cache without another miss provider");
    synchronize();
    strata_device_cache = std::make_unique<strata::deepseek_v4::DeviceExpertCache>(
        model, *moe_cache, sched.get(), strata_staging.get());
    moe_cache->strata_set_cache_hooks(strata::deepseek_v4::DeviceExpertCache::slot_map,
        strata::deepseek_v4::DeviceExpertCache::decorate, strata::deepseek_v4::DeviceExpertCache::guard,
        strata_device_cache.get());
}
void llama_context::strata_prepare_device_cache(size_t tokens) {
    if (!strata_device_cache) return;
    const char* phase_grow=std::getenv("STRATA_DSV4_TG_CACHE_GROW_MIB");
    if (strata::deepseek_v4::native_tg_cpu_only()&&phase_grow&&std::atoi(phase_grow)>0&&strata_device_cache->phase_change((tokens==1||(tokens==2&&strata::deepseek_v4::native_tg_cpu_only())))) {
        synchronize();
        gf_res_prev_active=nullptr;
        for(auto& graph:gf_res_prev)if(graph)graph->reset();
        const char* grow=std::getenv("STRATA_DSV4_TG_CACHE_GROW_MIB");
        if(grow&&std::atoi(grow)>0)ggml_backend_sched_release_buffers(sched.get());
    }
    strata_device_cache->prepare((tokens == 1 || (tokens == 2 && strata::deepseek_v4::native_tg_cpu_only())));
}
void llama_context::strata_enable_device_cache(bool enabled) {
    if (!strata_device_cache) throw std::runtime_error("context has no device cache");
    synchronize();
    ggml_backend_sched_reset(sched.get());
    gf_res_prev_active = nullptr;
    for (auto& graph : gf_res_prev) if (graph) graph->reset();
    strata_device_cache->set_enabled(enabled);
}
void llama_context::strata_invalidate_graph_for_fusion_switch() {
    synchronize();
    ggml_backend_sched_reset(sched.get());
    gf_res_prev_active = nullptr;
    for (auto& graph : gf_res_prev) if (graph) graph->reset();
}
void llama_context::strata_verify_device_cache() const {
    if (strata_device_cache) strata_device_cache->verify();
}
strata::deepseek_v4::DeviceCacheMetrics llama_context::strata_device_cache_metrics() const {
    return strata_device_cache ? strata_device_cache->metrics() : strata::deepseek_v4::DeviceCacheMetrics{};
}
namespace strata::deepseek_v4 {
void configure_device_expert_cache(llama_context* context) { context->strata_set_device_cache(); }
void prepare_device_expert_cache(llama_context* context, size_t tokens) { context->strata_prepare_device_cache(tokens); }
void enable_device_expert_cache(llama_context* context, bool enabled) { context->strata_enable_device_cache(enabled); }
void verify_device_expert_cache(llama_context* context) { context->strata_verify_device_cache(); }
DeviceCacheMetrics device_expert_cache_metrics(llama_context* context) { return context->strata_device_cache_metrics(); }
}

void llama_context::strata_idle_device_cache() {if(strata_device_cache)strata_device_cache->idle();}
namespace strata::deepseek_v4 {
void idle_device_expert_cache(llama_context* context){context->strata_idle_device_cache();}
}

namespace strata::deepseek_v4 {
ggml_backend_t device_expert_backend(llama_context* context) {
    auto* sched=context->get_sched();
    for(int i=0;i<ggml_backend_sched_get_n_backends(sched);++i){auto* backend=ggml_backend_sched_get_backend(sched,i);
        auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        if(reg&&std::strcmp(ggml_backend_reg_name(reg),"CUDA")==0)return backend;}
    return nullptr;
}
}

void llama_context::strata_write_routing_seed(const std::string& path) const {if(strata_device_cache)strata_device_cache->write_routing_seed(path);}
namespace strata::deepseek_v4 {
void write_device_routing_seed(llama_context* context,const std::string& path){context->strata_write_routing_seed(path);}
}

strata::deepseek_v4::Mxfp4CpuService& llama_context::strata_cpu_service() {
    if(!strata_device_cache)throw std::runtime_error("native expert authority missing");return strata_device_cache->cpu_service();
}
void llama_context::strata_capture_nextn_tail(size_t rows) {
    if(rows>128||model.arch!=LLM_ARCH_DEEPSEEK4)throw std::runtime_error("Strata hidden tail request invalid");
    synchronize();strata_nextn_rows=rows;strata_nextn_values.resize(rows*16384);
}
const std::vector<float>& llama_context::strata_nextn_tail() {synchronize();return strata_nextn_values;}
