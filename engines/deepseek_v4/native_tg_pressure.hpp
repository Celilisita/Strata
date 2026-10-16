#pragma once
#include "native_tg_graph.hpp"
#include "device_cache.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "pp_stream_probe_graph.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::deepseek_v4 {
// Exercise the real service with a full-size bank and a long enqueue graph.
// All layer identities borrow authenticated layer-zero weights; this tests
// scheduling/lifetime, not 43-layer model quality or end-to-end performance.
inline bool native_tg_pressure_probe(llama_model& model, ggml_backend_dev_t device,
    const std::array<ggml_tensor*,3>& source, const float* norm, std::ostream& out) {
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
    if (!backend || !cpu) throw std::runtime_error("native pressure backends unavailable");
    std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
    std::vector<ggml_backend_buffer_type_t> bufts{ggml_backend_get_default_buffer_type(backend.get()),ggml_backend_cpu_buffer_type()};
    std::unique_ptr<ggml_backend_sched,decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends.data(),bufts.data(),2,16384,false,true),ggml_backend_sched_free);
    llama_moe_cache cache(model,backends,bufts,16384ull<<20);
    cache.strata_set_leaf_aliases();
    ExpertStaging staging(backend.get());
    staging.register_sources(std::vector<ggml_tensor*>(source.begin(),source.end()),3422552064ull);
    // Materialize the real PP pipeline's high-water scratch in this same
    // backend, then retain enough GPU ballast to reproduce tight headroom.
    PpProbeGraph pp(backend.get(),source,8192,model.hparams.swiglu_clamp_exp[0],true,32);
    std::vector<float> pp_x(8192*4096), pp_w(8192*6,.25f);
    std::vector<int32_t> pp_ids(8192*6);
    for(int token=0;token<8192;++token) {
        std::copy_n(norm,4096,pp_x.data()+token*4096);
        for(int rank=0;rank<6;++rank)pp_ids[token*6+rank]=(token*17+rank*43)%256;
    }
    pp.inputs(pp_x,pp_ids,pp_w);
    DeviceExpertCache service(model,cache,scheduler.get(),&staging);
    service.prepare(true); // Decorate native graph nodes while the service is active.
    auto* raw=ggml_init({16384*ggml_tensor_overhead()+ggml_graph_overhead_custom(16384,false),nullptr,true});
    if(!raw) throw std::runtime_error("native pressure graph metadata unavailable");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> meta(raw,ggml_free);
    auto* parent=ggml_new_tensor_2d(raw,GGML_TYPE_F32,4096,2);
    auto* input=ggml_view_2d(raw,parent,4096,1,parent->nb[1],0);
    auto* activation_view=input;
    std::array<ggml_tensor*,43> ids{};
    auto* weights=ggml_new_tensor_3d(raw,GGML_TYPE_F32,1,6,1);
    auto* graph=ggml_new_graph_custom(raw,16384,false);
    for(int layer=0;layer<43;++layer) {
        if(layer==2) for(int queued=0;queued<5000;++queued) input=ggml_rms_norm(raw,input,1e-6f);
        // Preserve the dependency and any NaN, but keep real nonzero FFN
        // activations at every layer. Repeatedly applying one aliased MLP
        // would otherwise shrink into cheap zero-activation CPU work.
        if(layer)input=ggml_add(raw,activation_view,ggml_scale(raw,input,0.0f));
        input=ggml_rms_norm(raw,input,1e-6f);
        ids[layer]=ggml_new_tensor_2d(raw,GGML_TYPE_I32,6,1);
        auto* map=DeviceExpertCache::slot_map(layer,1,6,cache.strata_directory(layer),&service);
        auto* lookup=ggml_get_rows(raw,map,ggml_reshape_1d(raw,ids[layer],6));
        DeviceExpertCache::decorate(lookup,layer,&service);
        const auto& l=model.layers[layer];
        input=build_native_tg(raw,input,ids[layer],weights,ggml_reshape_2d(raw,lookup,6,1),
            cache.get_experts(l.ffn_gate_exps),cache.get_experts(l.ffn_up_exps),cache.get_experts(l.ffn_down_exps),
            l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps,model.hparams.swiglu_clamp_exp[layer],layer);
        if(!input) throw std::runtime_error("native pressure dispatch unavailable");
        if(native_tg_cpu_only()) {
            auto* shared=ggml_scale(raw,activation_view,.125f);
            auto* joined=finish_native_tg(raw,input,shared);
            if(!joined)throw std::runtime_error("CPU-only pressure join missing");
            input=joined;
        }
    }
    ggml_set_output(input);ggml_build_forward_expand(graph,input);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(raw,backend.get()),ggml_backend_buffer_free);
    if(!buffer)throw std::runtime_error("native pressure activation storage unavailable");
    service.prepare(false);
    if(ggml_backend_graph_compute_async(backend.get(),pp.graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("native pressure PP setup failed");
    ggml_backend_synchronize(backend.get());
    size_t free_before=0,total=0;
    ggml_backend_dev_memory(device,&free_before,&total);
    constexpr size_t margin=512ull<<20;
    if(free_before<=margin)throw std::runtime_error("native pressure initial memory budget exhausted");
    const size_t ballast_bytes=free_before-margin;
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> ballast(
        ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend.get()),ballast_bytes),ggml_backend_buffer_free);
    if(!ballast)throw std::runtime_error("native pressure GPU ballast unavailable");
    ggml_backend_buffer_clear(ballast.get(),0);ggml_backend_synchronize(backend.get());
    size_t free_with_pp=0,free_after_retire=0;
    ggml_backend_dev_memory(device,&free_with_pp,&total);
    const auto cache_before_tg=service.metrics().capacity;
    service.prepare(true);
    const auto cache_after_tg=service.metrics().capacity;
    const auto expanded_cache=cache_after_tg.backend_alloc_bytes-cache_before_tg.backend_alloc_bytes;
    ggml_backend_dev_memory(device,&free_after_retire,&total);
    std::vector<float> initial(norm,norm+4096);initial.insert(initial.end(),norm,norm+4096);
    ggml_backend_tensor_set(parent,initial.data(),0,initial.size()*sizeof(float));
    const std::array<float,6> w{.1f,.15f,.2f,.25f,.3f,.5f};
    ggml_backend_tensor_set(weights,w.data(),0,sizeof(w));
    bool passed=free_after_retire+expanded_cache>free_with_pp+(1ull<<30);const auto before=service.metrics();
    out<<"{\"model_inference\":false,\"real_layer0_weights\":true,\"layer_identities\":43,\"cache_mib\":16384,\"pressure_norm_nodes\":5000,\"cases\":[";
    for(int test=0;test<7;++test) {
        // Change a real source pointer after captured replay to exercise eager
        // graph warmup reset, which occurred at the whole-model failure.
        activation_view->data=static_cast<char*>(parent->data)+(test>=3?4096*sizeof(float):0);
        for(int layer=0;layer<43;++layer) {
            std::array<int32_t,6> route{};
            for(int rank=0;rank<6;++rank)route[rank]=(test*19+layer*7+rank*37)%256;
            ggml_backend_tensor_set(ids[layer],route.data(),0,sizeof(route));
        }
        if(ggml_backend_graph_compute_async(backend.get(),graph)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("native pressure compute failed");
        ggml_backend_synchronize(backend.get());service.verify();
        for(int idle=0;idle<16;++idle)service.idle();
        std::array<float,4096> result{};ggml_backend_tensor_get(input,result.data(),0,sizeof(result));
        const bool finite=std::all_of(result.begin(),result.end(),[](float v){return std::isfinite(v);});passed&=finite;
        if(test)out<<',';out<<"{\"test\":"<<test<<",\"finite\":"<<(finite?"true":"false")<<'}';out.flush();
    }
    const auto after=service.metrics();
    passed&=after.gpu[0]-before.gpu[0]==7*43&&!after.capacity.invalid_directory_links&&after.gpu[4]==before.gpu[4];
    out<<"],\"lookups\":"<<after.gpu[0]-before.gpu[0]<<",\"payload_bytes\":"<<after.transport.bytes-before.transport.bytes
       <<",\"cpu_routes\":"<<after.cpu_routes-before.cpu_routes<<",\"host_services\":"<<after.host_services-before.host_services
       <<",\"timeouts\":"<<after.gpu[4]-before.gpu[4]<<",\"graph_preuploads\":"<<after.dma_costs.graph_preuploads-before.dma_costs.graph_preuploads
       <<",\"enqueue_budget_records\":"<<after.dma_costs.enqueue_budget_records-before.dma_costs.enqueue_budget_records
       <<",\"ballast_bytes\":"<<ballast_bytes<<",\"free_bytes_with_pp_scratch\":"<<free_with_pp
       <<",\"free_bytes_after_retire\":"<<free_after_retire<<",\"pp_scratch_retired\":"<<(free_after_retire+expanded_cache>free_with_pp+(1ull<<30)?"true":"false")
       <<",\"expanded_cache_bytes\":"<<expanded_cache<<",\"background_weight_bytes\":"<<after.background_bytes-before.background_bytes<<",\"background_commits\":"<<after.background_commits-before.background_commits
       <<",\"passed\":"<<(passed?"true":"false")<<'}';
    return passed;
}
}
