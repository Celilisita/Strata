#pragma once
#include "pp_stream_graph.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "expert_staging.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <fstream>
#include "cache_cost_audit.hpp"
#include "expert_frequency.hpp"
#include <cstring>
#include <numeric>
namespace strata::deepseek_v4 {
// Distinct GPU rows in all43 layers exercise the full cache working set.
// Host payloads intentionally alias one immutable real layer: no trunk load.
inline bool pp_working_set_probe(llama_model& model,llama_moe_cache& cache,ggml_backend_t backend,
        PpProbeGraph& graph,ExpertStaging& upload,const std::vector<float>& expected,
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>& ballast,std::ostream& out,const std::vector<float>& x,const std::vector<float>& weights,PpProbeGraph* reference) {
    auto fill=[&](int layer,const std::array<int32_t,6>& ids){cache.strata_fill_selected(layer,ids.data(),6,ExpertStaging::dispatch,&upload);};
    for(int layer=0;layer<43;++layer)for(int group=0;group<5;++group){std::array<int32_t,6> ids;
        for(int rank=0;rank<6;++rank)ids[rank]=(layer*7+group*6+rank)%256;fill(layer,ids);}
    ggml_backend_synchronize(backend);const int capacity=int(cache.strata_cost_snapshot().permanent_slots);
    auto sources=pp_stream_sources(graph.out);bool passed=true;
    const char* route_path=std::getenv("STRATA_DSV4_PROBE_PP_ROUTING_FILE");std::vector<int32_t> recorded;
    const size_t routes_per_layer=size_t(ggml_nelements(graph.ids));
    if(route_path){if(!reference)throw std::runtime_error("Observed histogram requires same-input reference");recorded.resize(43*routes_per_layer);
        std::ifstream input(route_path,std::ios::binary);input.read(reinterpret_cast<char*>(recorded.data()),recorded.size()*sizeof(int32_t));
        if(!input||input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Observed PP histogram fixture size differs");}
    out<<"{\"scope\":\"distinct43-layer GPU cache working set, identical real layer0 host payload aliases; no trunk model loaded\",\"permanent_slots\":"<<capacity<<",\"phases\":[";
    const bool staging_aba=std::getenv("STRATA_DSV4_PROBE_PP_STAGING_ABA")&&
        !std::strcmp(std::getenv("STRATA_DSV4_PROBE_PP_STAGING_ABA"),"1");
    const bool direct_aba=std::getenv("STRATA_DSV4_PROBE_PP_DIRECT_ABA")&&
        !std::strcmp(std::getenv("STRATA_DSV4_PROBE_PP_DIRECT_ABA"),"1");
    const bool compare=staging_aba||direct_aba;
    for(int phase=0;phase<(compare?3:2);++phase) {
        if(direct_aba) {
#ifdef _WIN32
            if(_putenv_s("STRATA_DSV4_PP_DIRECT_CACHE_GROUPS",phase==1?"1":"0"))
                throw std::runtime_error("PP direct cache mode switch failed");
#else
            if(setenv("STRATA_DSV4_PP_DIRECT_CACHE_GROUPS",phase==1?"1":"0",1))
                throw std::runtime_error("PP direct cache mode switch failed");
#endif
        }
        if(staging_aba) {
#ifdef _WIN32
            if(_putenv_s("STRATA_DSV4_PP_STAGING_SLABS",phase==1?"3":"1"))
                throw std::runtime_error("PP staging mode switch failed");
#else
            if(setenv("STRATA_DSV4_PP_STAGING_SLABS",phase==1?"3":"1",1))
                throw std::runtime_error("PP staging mode switch failed");
#endif
        }
        if(phase&&!compare) {
            // Match a PP->TG->PP cycle: expand, fill new rows, compact, release.
            ballast.reset();cache.strata_resize_phase(capacity+288,ExpertStaging::dispatch,&upload);
            int remaining=288;
            for(int layer=0;layer<43&&remaining; ++layer){auto& l=model.layers[layer];const auto* view=cache.strata_pp_view(layer,l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps);
                std::vector<int32_t> absent;const auto* map=static_cast<const int32_t*>(view->directory->data);
                for(int id=0;id<256;++id)if(map[id]<0)absent.push_back(id);
                for(size_t i=0;i+6<=absent.size()&&remaining;i+=6){std::array<int32_t,6> ids;std::copy_n(absent.data()+i,6,ids.begin());fill(layer,ids);remaining-=6;}}
            if(remaining)throw std::runtime_error("Working set fixture cannot fill growth rows");
            std::vector<uint32_t> frequency(43*256+1);for(size_t key=0;key+1<frequency.size();++key)frequency[key]=1+uint32_t((key*17)%101);
            if(auto* counter=cache.strata_frequency())counter->import_state(frequency.data(),frequency.size());
            cache.strata_resize_phase(capacity,ExpertStaging::dispatch,&upload);ggml_backend_synchronize(backend);
            size_t free=0,total=0;ggml_backend_dev_memory(ggml_backend_get_device(backend),&free,&total);
            constexpr size_t margin=512ull<<20;if(free<=margin)throw std::runtime_error("Working set pressure has no margin");
            ballast.reset(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend),free-margin));
            if(!ballast)throw std::runtime_error("Working set ballast allocation failed");ggml_backend_buffer_clear(ballast.get(),0);ggml_backend_synchronize(backend);
        }
        const auto before=pp_stream_backend_metrics(backend);double squared=0,norm=0,max_abs=0;
        auto start=std::chrono::steady_clock::now();
        for(int layer=0;layer<43;++layer) {
            const char* active=std::getenv("STRATA_DSV4_PROBE_PP_ACTIVE_BALLAST");
            if(active&&!std::strcmp(active,"1")){ggml_backend_buffer_clear(ballast.get(),0);ggml_backend_synchronize(backend);}
            auto& l=model.layers[layer];sources.gate=l.ffn_gate_exps;sources.up=l.ffn_up_exps;sources.down=l.ffn_down_exps;
            sources.cache=cache.strata_pp_view(layer,l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps);
            // The bounded PP graph never captures CUDA execution. All source
            // identities and directory ownership change together at quiescence.
            std::memcpy(reinterpret_cast<char*>(graph.out->op_params)+pp_stream_params_offset,&sources,sizeof(sources));
            std::vector<int32_t> route;
            if(route_path){route.assign(recorded.begin()+layer*routes_per_layer,recorded.begin()+(layer+1)*routes_per_layer);graph.inputs(x,route,weights);}
            if(ggml_backend_graph_compute_async(backend,graph.graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Working set PP compute failed");
            ggml_backend_synchronize(backend);
            if(layer==0||layer==21||layer==42){auto actual=graph.output();std::vector<float> checked;
                if(route_path){reference->inputs(x,route,weights);if(ggml_backend_graph_compute_async(backend,reference->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Observed histogram reference failed");ggml_backend_synchronize(backend);checked=reference->output();}
                const auto& baseline=route_path?checked:expected;for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i]))throw std::runtime_error("Working set nonfinite output");
                const double e=double(actual[i])-baseline[i];squared+=e*e;norm+=double(baseline[i])*baseline[i];max_abs=std::max(max_abs,std::abs(e));}}
        }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();const auto after=pp_stream_backend_metrics(backend);
        const auto audit=cache.strata_cost_snapshot();const double relative=std::sqrt(squared/std::max(norm,1e-30));
        const bool ok=relative<=1e-4&&max_abs<=.01&&!audit.invalid_directory_links&&after.layers-before.layers==43;passed&=ok;
        if(phase)out<<',';out<<"{\"after_grow_shrink\":"<<(phase&&!compare?"true":"false")
            <<",\"wall_ms_including3_output_checks\":"<<ms
            <<",\"direct_cache_groups\":"<<(direct_aba&&phase==1?"true":"false")
            <<",\"staging_slabs\":"<<(staging_aba&&phase==1?3:1)
            <<",\"h2d_bytes\":"<<after.payload_bytes-before.payload_bytes<<",\"gpu_reused_bytes\":"<<after.cache_reused_bytes-before.cache_reused_bytes
            <<",\"gpu_layer_ns\":"<<after.gpu_layer_ns-before.gpu_layer_ns
            <<",\"gpu_prelude_ns\":"<<after.gpu_prelude_ns-before.gpu_prelude_ns
            <<",\"gpu_intergroup_ns\":"<<after.gpu_intergroup_ns-before.gpu_intergroup_ns
            <<",\"gpu_weight_wait_ns\":"<<after.gpu_weight_wait_ns-before.gpu_weight_wait_ns
            <<",\"gpu_products_span_ns\":"<<after.gpu_products_ns-before.gpu_products_ns
            <<",\"cache_copy_calls\":"<<after.cache_copy_calls-before.cache_copy_calls
            <<",\"staged_bytes\":"<<after.staged_bytes-before.staged_bytes
            <<",\"staging_wait_ns\":"<<after.staging_wait_ns-before.staging_wait_ns
            <<",\"source_cpu_copy_ns\":"<<after.source_cpu_copy_ns-before.source_cpu_copy_ns
            <<",\"relative_rms\":"<<relative<<",\"max_abs\":"<<max_abs<<",\"passed\":"<<(ok?"true":"false")<<'}';out.flush();
    }
    out<<"],\"passed\":"<<(passed?"true":"false")<<'}';return passed;
}
}
