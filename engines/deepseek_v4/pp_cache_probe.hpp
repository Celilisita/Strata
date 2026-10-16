#pragma once
#include "pp_stream_probe_graph.hpp"
#include "pp_stream_metrics.hpp"
#include "pp_cache_view.hpp"
#include "expert_staging.hpp"
#include "pp_working_set_probe.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <ostream>
#include <string>
#include <stdexcept>

namespace strata::deepseek_v4 {
inline bool pp_cache_operator_probe(const std::string& path, ggml_backend_t backend,
    const std::array<ggml_tensor*,3>& source, const std::vector<float>& x,
    const std::vector<int32_t>& ids, const std::vector<float>& weights,
    const std::vector<float>& expected, float clamp, int window, std::ostream& out,PpProbeGraph* reference=nullptr) {
    const auto device=ggml_backend_get_device(backend);
    ggml_backend_dev_t devices[]{device,nullptr};
    const llama_model_tensor_buft_override overrides[]{
        {"blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    auto mp=llama_model_default_params();mp.devices=devices;mp.tensor_buft_overrides=overrides;
    mp.n_gpu_layers=-1;mp.use_extra_bufts=false;mp.load_mtp=false;
    mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.no_alloc=true;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(path.c_str(),mp),llama_model_free);
    if(!model||model->layers.size()!=43)throw std::runtime_error("PP cache metadata model unavailable");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> meta(ggml_init({129*ggml_tensor_overhead(),nullptr,true}),ggml_free);
    if(!meta)throw std::runtime_error("PP cache source identities unavailable");
    for(int layer=0;layer<43;++layer) {
        std::array<ggml_tensor*,3> aliases{};
        for(int role=0;role<3;++role) {
            aliases[role]=ggml_new_tensor_3d(meta.get(),GGML_TYPE_MXFP4,source[role]->ne[0],source[role]->ne[1],256);
            aliases[role]->data=source[role]->data;aliases[role]->buffer=source[role]->buffer;
        }
        auto& l=model->layers[layer];l.ffn_gate_exps=aliases[0];l.ffn_up_exps=aliases[1];l.ffn_down_exps=aliases[2];l.ffn_gate_up_exps=nullptr;
    }
    model->hparams.no_alloc=false;
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
    if(!cpu)throw std::runtime_error("PP cache CPU metadata backend unavailable");
    std::vector<ggml_backend_t> backends{backend,cpu.get()};
    std::vector<ggml_backend_buffer_type_t> bufts{ggml_backend_get_default_buffer_type(backend),ggml_backend_cpu_buffer_type()};
    const char* pressure_env=std::getenv("STRATA_DSV4_PROBE_PP_CACHE_PRESSURE");
    const bool pressure=pressure_env&&std::string(pressure_env)=="1";
    llama_moe_cache cache(*model,backends,bufts,pressure?16384ull<<20:263ull*13369344);
    cache.strata_set_leaf_aliases();
    auto& l=model->layers[0];
    const std::array<ggml_tensor*,3> aliases{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps};
    const auto* view=cache.strata_pp_view(0,aliases[0],aliases[1],aliases[2]);
    if(!view)throw std::runtime_error("PP authoritative cache projection unavailable");
    PpProbeGraph candidate(backend,aliases,int(ids.size()/6),clamp,true,window,view);
    candidate.inputs(x,ids,weights);
    std::array<uint32_t,256> expected_hist{};for(int id:ids)++expected_hist.at(id);
    // Fixture cache fills borrow the real mmap source. A DMA span crossing a
    // registered/pageable boundary is invalid on this backend; use the same
    // bounded pinned upload service as production, outside timed execution.
    ExpertStaging cache_upload(backend);
    auto upload=ExpertStaging::dispatch;
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> ballast(nullptr,ggml_backend_buffer_free);
    size_t ballast_bytes=0;
    if(pressure) {
        size_t free=0,total=0;ggml_backend_dev_memory(device,&free,&total);
        constexpr size_t margin=512ull<<20;
        if(free<=margin)throw std::runtime_error("PP pressure fixture has no headroom");
        ballast_bytes=free-margin;
        ballast.reset(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend),ballast_bytes));
        if(!ballast)throw std::runtime_error("PP pressure ballast allocation failed");
        ggml_backend_buffer_clear(ballast.get(),0);ggml_backend_synchronize(backend);
    }
    const char* sweep=std::getenv("STRATA_DSV4_PROBE_PP_LAYER_SWEEP");
    if(sweep&&std::string(sweep)=="1")return pp_working_set_probe(*model,cache,backend,candidate,cache_upload,expected,ballast,out,x,weights,reference);
    bool passed=true;
    out<<"{\"scope\":\"real layer-zero payload; authoritative cache reuse only\",\"ballast_bytes\":"<<ballast_bytes<<",\"cases\":[";
    for(int phase=0;phase<3;++phase) {
        if(phase==1) {
            const std::array<int32_t,6> a{0,5,17,60,103,146},b{189,232,250,253,254,255};
            cache.strata_fill_selected(0,a.data(),6,upload,&cache_upload);
            cache.strata_fill_selected(0,b.data(),6,upload,&cache_upload);
        } else if(phase==2) {
            for(int first=0;first<256;first+=6) {
                std::array<int32_t,6> route{};
                for(int rank=0;rank<6;++rank)route[rank]=(first+rank)%256;
                cache.strata_fill_selected(0,route.data(),6,upload,&cache_upload);
            }
        }
        ggml_backend_synchronize(backend);
        const auto before=pp_stream_backend_metrics(backend);
        const auto start=std::chrono::steady_clock::now();
        if(ggml_backend_graph_compute_async(backend,candidate.graph)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("PP cached execution failed");
        ggml_backend_synchronize(backend);
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const auto after=pp_stream_backend_metrics(backend);
        const auto actual=candidate.output();
        if(*view->prefill_tokens!=int(ids.size()/6)||!std::equal(expected_hist.begin(),expected_hist.end(),view->prefill_counts))throw std::runtime_error("actual PP route histogram differs");
        double error=0,reference=0,max_abs=0;size_t nonfinite=0;
        for(size_t row=0;row<actual.size();++row) {
            if(!std::isfinite(actual[row])){++nonfinite;continue;}
            const double e=double(actual[row])-expected[row];error+=e*e;reference+=double(expected[row])*expected[row];max_abs=std::max(max_abs,std::abs(e));
        }
        const uint64_t reused=after.cache_reused_bytes-before.cache_reused_bytes;
        const uint64_t payload=after.payload_bytes-before.payload_bytes;
        const uint64_t count=phase==0?0:phase==1?12:256;
        const double relative=std::sqrt(error/std::max(reference,1e-30));
        const bool ok=!nonfinite&&relative<=1e-4&&max_abs<=.01&&reused==count*13369344&&payload==(256-count)*13369344;
        passed&=ok;
        if(phase)out<<',';
        out<<"{\"resident_experts\":"<<count<<",\"ms\":"<<ms<<",\"h2d_bytes\":"<<payload<<",\"reused_gpu_bytes\":"<<reused
           <<",\"relative_rms\":"<<relative<<",\"max_abs\":"<<max_abs<<",\"passed\":"<<(ok?"true":"false");
        if(phase==2) {
            auto set_mode=[](const char* mode) {
#ifdef _WIN32
                if(_putenv_s("STRATA_DSV4_PP_RAGGED_MMQ",mode))throw std::runtime_error("PP ragged mode update failed");
#else
                if(setenv("STRATA_DSV4_PP_RAGGED_MMQ",mode,1))throw std::runtime_error("PP ragged mode update failed");
#endif
            };
            const char* previous=std::getenv("STRATA_DSV4_PP_RAGGED_MMQ");
            const std::string restore=previous?previous:"1";
            std::array<std::array<double,3>,2> times{};
            const auto ab_before=pp_stream_backend_metrics(backend);
            for(int mode=0;mode<2;++mode) {
                set_mode(mode?"1":"0");
                for(int repeat=0;repeat<4;++repeat) {
                    const auto begin=std::chrono::steady_clock::now();
                    if(ggml_backend_graph_compute_async(backend,candidate.graph)!=GGML_STATUS_SUCCESS)
                        throw std::runtime_error("PP ragged resident compute failed");
                    ggml_backend_synchronize(backend);
                    if(repeat)times[mode][repeat-1]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                }
                const auto check=candidate.output();double squared=0,base=0;size_t invalid=0;
                for(size_t row=0;row<check.size();++row) {
                    if(!std::isfinite(check[row])){++invalid;continue;}
                    const double e=double(check[row])-expected[row];squared+=e*e;base+=double(expected[row])*expected[row];
                }
                passed&=!invalid&&std::sqrt(squared/std::max(base,1e-30))<=1e-4;
            }
            set_mode(restore.c_str());
            const auto ab_after=pp_stream_backend_metrics(backend);
            auto control=times[0],ragged=times[1];std::sort(control.begin(),control.end());std::sort(ragged.begin(),ragged.end());
            const bool competitive=ragged[1]<=control[1]*1.05&&ab_after.payload_bytes==ab_before.payload_bytes;
            passed&=competitive;
            out<<",\"resident_compute_ab\":{\"control_ms\":["<<times[0][0]<<','<<times[0][1]<<','<<times[0][2]
               <<"],\"ragged_ms\":["<<times[1][0]<<','<<times[1][1]<<','<<times[1][2]
               <<"],\"speedup\":"<<control[1]/ragged[1]<<",\"no_h2d\":"<<(ab_after.payload_bytes==ab_before.payload_bytes?"true":"false")
               <<",\"competitive\":"<<(competitive?"true":"false")<<'}';
        }
        out<<'}';out.flush();
    }
    out<<"],\"passed\":"<<(passed?"true":"false")<<",\"end_to_end_target_met\":false}";
    return passed;
}
}
