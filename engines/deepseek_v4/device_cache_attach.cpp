#include "device_cache.hpp"
#include "phase_allocator.hpp"
#include "cuda_pool_retirement.hpp"
#include "native_tg_graph.hpp"
#include "../glm5_next/phase_options.hpp"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-moe-cache.h"
#include "llama-model.h"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include "../glm5_next/verify4_options.hpp"

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
    if (strata_device_cache->refresh_cold_gpu_rows()) {
        synchronize();ggml_backend_sched_reset(sched.get());gf_res_prev_active=nullptr;
        for(auto& graph:gf_res_prev)if(graph)graph->reset();
    }
    const char* phase_grow=std::getenv("STRATA_DSV4_TG_CACHE_GROW_MIB");
    const bool glm_phase=model.arch==LLM_ARCH_GLM5_NEXT && strata::glm5_next::phase_cache_requested();
    const bool tg=tokens==1||(tokens==2&&strata::deepseek_v4::native_tg_cpu_only())||
        (model.arch==LLM_ARCH_GLM5_NEXT&&tokens==4&&strata::glm5_next::verify4_probe_requested());
    if ((glm_phase || (strata::deepseek_v4::native_tg_cpu_only()&&phase_grow&&std::atoi(phase_grow)>0))&&strata_device_cache->phase_change(tg)) {
        synchronize();
        gf_res_prev_active=nullptr;
        for(auto& graph:gf_res_prev)if(graph)graph->reset();
        const char* grow=std::getenv("STRATA_DSV4_TG_CACHE_GROW_MIB");
        if(glm_phase || (grow&&std::atoi(grow)>0))ggml_backend_sched_release_buffers(sched.get());
    }
    strata_device_cache->prepare(tg);
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
void llama_context::strata_begin_pp_plan_receipt(uint64_t nonce,bool enabled) {
    if(!strata_device_cache)throw std::runtime_error("PP receipt requires the existing device cache");
    strata_device_cache->begin_pp_plan_receipt(nonce,enabled);
}
const strata::deepseek_v4::PpPlanReceiptPod* llama_context::strata_read_pp_plan_receipt() const {
    if(!strata_device_cache)throw std::runtime_error("PP receipt requires the existing device cache");
    return strata_device_cache->read_pp_plan_receipt();
}
void llama_context::strata_begin_route_capture() {if(!strata_device_cache)throw std::runtime_error("Route capture has no cache owner");strata_device_cache->begin_route_capture();}
void llama_context::strata_write_route_capture(const std::string& path) {if(!strata_device_cache)throw std::runtime_error("Route capture has no cache owner");strata_device_cache->write_route_capture(path);}
namespace strata::deepseek_v4 {
void write_device_routing_seed(llama_context* context,const std::string& path){context->strata_write_routing_seed(path);}
void begin_device_pp_plan_receipt(llama_context* context,uint64_t nonce,bool enabled){context->strata_begin_pp_plan_receipt(nonce,enabled); }
const PpPlanReceiptPod* read_device_pp_plan_receipt(llama_context* context){return context->strata_read_pp_plan_receipt();}
void begin_device_route_capture(llama_context* context){context->strata_begin_route_capture();}
void write_device_route_capture(llama_context* context,const std::string& path){context->strata_write_route_capture(path);}
}

strata::deepseek_v4::Mxfp4CpuService& llama_context::strata_cpu_service() {
    if(!strata_device_cache)throw std::runtime_error("native expert authority missing");return strata_device_cache->cpu_service();
}
void llama_context::strata_capture_nextn_tail(size_t rows) {
    if(rows>128||model.arch!=LLM_ARCH_DEEPSEEK4)throw std::runtime_error("Strata hidden tail request invalid");
    synchronize();strata_nextn_rows=rows;strata_nextn_values.resize(rows*16384);
}
const std::vector<float>& llama_context::strata_nextn_tail() {synchronize();return strata_nextn_values;}

void llama_context::strata_reverse_phase_probe_prior() {
    synchronize();if(!strata_device_cache)throw std::runtime_error("phase probe has no cache authority");
    strata_device_cache->phase_probe_reverse_prior();
}
namespace strata::deepseek_v4 {
void reverse_phase_probe_prior(llama_context* ctx){ctx->strata_reverse_phase_probe_prior();}
}

std::array<uint64_t,4> llama_context::strata_probe_background() {
    synchronize();if(!strata_device_cache)throw std::runtime_error("background probe has no cache authority");
    return strata_device_cache->probe_background();
}
namespace strata::deepseek_v4 {
std::array<uint64_t,4> probe_device_background(llama_context* ctx){return ctx->strata_probe_background();}
}

void llama_context::strata_settle_background() {if(strata_device_cache)strata_device_cache->settle_background();}
namespace strata::deepseek_v4 {
void settle_device_background(llama_context* ctx){ctx->strata_settle_background();}
}

ggml_backend_sched_t llama_context::strata_borrow_cache_scheduler(ggml_backend_sched_t scheduler) {
    if(!strata_device_cache)throw std::runtime_error("borrowed graph has no expert authority");
    return strata_device_cache->borrow_scheduler(scheduler);
}

void llama_context::strata_retire_compute_buffers() {
    synchronize();gf_res_prev_active=nullptr;
    for(auto& graph:gf_res_prev)if(graph)graph->reset();
    ggml_backend_sched_release_buffers(sched.get());
    auto* backend=strata::deepseek_v4::device_expert_backend(this);
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using Retire=bool(*)(ggml_backend_t);
    auto retire=reinterpret_cast<Retire>(ggml_backend_reg_get_proc_address(reg,"strata_pp_stream_enter_tg"));
    if(!retire||!retire(backend))throw std::runtime_error("NextN prefix warm requires retired primary PP workspace");
    if(model.arch==LLM_ARCH_GLM5_NEXT) {
        using strata::deepseek_v4::CudaPoolRetire;
        strata::deepseek_v4::CudaPoolRetirement state;
        auto retire_pool=reinterpret_cast<CudaPoolRetire>(ggml_backend_reg_get_proc_address(reg,"strata_cuda_pool_retire"));
        if(!retire_pool||!retire_pool(backend,&state))
            throw std::runtime_error("GLM PP graph resize requires a quiescent CUDA pool");
    }
}

void llama_context::strata_set_cache_graph_retirement(void(*fn)(void*),void* user) {
    if(!strata_device_cache){if(!fn)return;throw std::runtime_error("GLM borrowed graph registration lacks cache owner");}
    strata_device_cache->set_graph_retirement(fn,user);
}
namespace strata::deepseek_v4 {
void register_device_cache_graph_retirement(llama_context* context,void(*fn)(void*),void* user) {
    if(context)context->strata_set_cache_graph_retirement(fn,user);
}
}
