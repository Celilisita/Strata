#pragma once
#include "mtp_drafter.hpp"
#include "device_cache.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "ggml-cpu.h"
#include "expert_frequency.hpp"
#include "pp_cache_view.hpp"
#include "mtp_main_kv_probe.hpp"
#include "mtp_context_probe.hpp"
#include <memory>
#include <ostream>
#include <cstdlib>
#include <cstdio>
namespace strata::deepseek_v4 {
// Production 44-layer cache identity, authentic MTP tensors, borrowed target
// embedding/head. Other layers contain metadata aliases only and are not run.
inline bool mtp_native_probe(const char* model_path,MtpSidecar& mtp,ggml_backend_t backend,
    ggml_tensor* embedding,ggml_tensor* head,const float* hidden,const int32_t* ids,std::ostream& out) {
    const auto mark=[](const char* stage){if(std::getenv("STRATA_DSV4_MTP_PROBE_MARKERS")){std::fprintf(stderr,"mtp_native_probe_stage=%s\n",stage);std::fflush(stderr);}};
    mark("start");
    const auto dev=ggml_backend_get_device(backend);ggml_backend_dev_t devices[]{dev,nullptr};
    const llama_model_tensor_buft_override overrides[]{{"blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    auto mp=llama_model_default_params();mp.devices=devices;mp.tensor_buft_overrides=overrides;mp.n_gpu_layers=-1;mp.main_gpu=0;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    mp.use_extra_bufts=false;mp.load_mtp=false;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.no_alloc=true;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path,mp),llama_model_free);
    if(!model||model->layers.size()!=43)throw std::runtime_error("MTP cache metadata fixture invalid");
    mark("model_metadata");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> meta(ggml_init({129*ggml_tensor_overhead(),nullptr,true}),ggml_free);
    const std::array<ggml_tensor*,3> source{mtp.tensor("blk.43.ffn_gate_exps.weight"),mtp.tensor("blk.43.ffn_up_exps.weight"),mtp.tensor("blk.43.ffn_down_exps.weight")};
    for(int layer=0;layer<43;++layer){std::array<ggml_tensor*,3> alias{};for(int r=0;r<3;++r){auto* t=source[r];alias[r]=ggml_new_tensor_3d(meta.get(),t->type,t->ne[0],t->ne[1],256);alias[r]->data=t->data;alias[r]->buffer=t->buffer;ggml_format_name(alias[r],"mtp_fixture.%d.%d",layer,r);}
        model->layers[layer].ffn_gate_exps=alias[0];model->layers[layer].ffn_up_exps=alias[1];model->layers[layer].ffn_down_exps=alias[2];model->layers[layer].ffn_gate_up_exps=nullptr;}
    model->strata_append_mtp_experts(source[0],source[1],source[2]);model->hparams.no_alloc=false;
    mtp_main_kv_probe(*model,backend);
    mtp_context_probe(*model,backend,embedding);
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
    std::vector<ggml_backend_t> backends{backend,cpu.get()};std::vector<ggml_backend_buffer_type_t> bufts{ggml_backend_get_default_buffer_type(backend),ggml_backend_cpu_buffer_type()};
    std::unique_ptr<ggml_backend_sched,decltype(&ggml_backend_sched_free)> scheduler(ggml_backend_sched_new(backends.data(),bufts.data(),2,4096,false,true),ggml_backend_sched_free);
    llama_moe_cache cache(*model,backends,bufts,19ull*13369344);cache.strata_set_leaf_aliases();
    mark("cache_metadata");
    for(int layer=0;layer<43;++layer){if(std::getenv("STRATA_DSV4_MTP_PROBE_MARKERS")){std::fprintf(stderr,"mtp_native_probe_layer=%d\n",layer);std::fflush(stderr);}
        const auto& l=model->layers[layer];auto* view=cache.strata_pp_view(layer,l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps);
        if(!view)throw std::runtime_error("MTP phase histogram fixture missing");*view->prefill_tokens=8192;for(int id=0;id<6;++id)view->prefill_counts[id]=8192;}
    mark("prefill_views_ready");
    cache.strata_use_prefill_frequency();std::vector<uint32_t> state(44*256+1);cache.strata_frequency()->export_state(state.data(),state.size());
    for(int layer=0;layer<44;++layer)for(int id=0;id<256;++id)if(state[layer*256+id]!=uint32_t(layer<43&&id<6?8192:0))throw std::runtime_error("MTP phase histogram projection differs");
    mark("frequency_ready");
    DeviceExpertCache service(*model,cache,scheduler.get(),nullptr);
    mark("device_service");
    cache.strata_set_cache_hooks(DeviceExpertCache::slot_map,DeviceExpertCache::decorate,DeviceExpertCache::guard,&service);service.prepare(true);
    mark("cache_prepared");
    MtpDrafter draft(mtp,backend,embedding,head,service.cpu_service(),1,nullptr,&cache);
    mark("baseline_drafter");
    const int32_t position=0;const auto before=service.metrics();draft.forward(hidden,ids,&position);service.verify();const auto cold=service.metrics();
    mark("baseline_cold");
    const auto set_cached_only=[](const char* value){
#ifdef _WIN32
        if(_putenv_s("STRATA_DSV4_MTP_CACHE_ONLY",value))throw std::runtime_error("MTP probe mode switch failed");
#else
        if(setenv("STRATA_DSV4_MTP_CACHE_ONLY",value,1))throw std::runtime_error("MTP probe mode switch failed");
#endif
    };
    set_cached_only("1");
    MtpDrafter cached_only(mtp,backend,embedding,head,service.cpu_service(),1,nullptr,&cache);
    mark("cached_only_drafter");
    set_cached_only("0");
    const auto skipped_before=service.metrics();const auto skipped_token=cached_only.forward(hidden,ids,&position).at(0);
    service.verify();const auto skipped=service.metrics();
    mark("cached_only_cold");
    auto route=draft.diagnostic_routes();service.set_enabled(true);
    for(int repeat=0;repeat<9000;++repeat)cache.strata_frequency()->observe(43,route.data(),6);
    cache.strata_fill_selected(43,route.data(),6,[](ggml_backend_t b,ggml_tensor* t,const void* p,size_t offset,size_t n,void*){ggml_backend_tensor_set_async(b,t,p,offset,n);return true;},nullptr);
    ggml_backend_synchronize(backend);service.prepare(true);draft.reset();cached_only.reset();
    const auto* directory=static_cast<const int32_t*>(cache.strata_directory(43)->data);
    for(int id:route)if(directory[id]<0)throw std::runtime_error("MTP all-hit probe could not admit selected route");
    mark("cache_filled");
    const auto hot_before=service.metrics();const auto hot_token=draft.forward(hidden,ids,&position).at(0);service.verify();
    const auto cached_hot_token=cached_only.forward(hidden,ids,&position).at(0);service.verify();const auto hot=service.metrics();
    mark("hot_pair");
    cached_only.reset();
    MtpStageMetrics steady_begin{},steady_end{};
    for(int i=0;i<144;++i){const int32_t at=i;cached_only.forward(hidden,ids,&at);
        if(i==127)steady_begin=cached_only.stage_metrics();}
    mark("steady_complete");
    steady_end=cached_only.stage_metrics();service.verify();
    const double steady_calls=double(steady_end.calls-steady_begin.calls);
    const auto scratch=draft.device_scratch_bytes(),dense=mtp.dense_device_bytes();
    const bool ok=cold.cpu_routes-before.cpu_routes==6&&cold.transport.bytes==before.transport.bytes&&
        skipped.cpu_routes==skipped_before.cpu_routes&&skipped.host_services==skipped_before.host_services&&
        skipped_token>=0&&hot_token==cached_hot_token&&hot.host_services==hot_before.host_services&&
        hot.cpu_routes==hot_before.cpu_routes&&hot.transport.bytes==hot_before.transport.bytes&&
        hot.gpu[1]-hot_before.gpu[1]==2&&dense+scratch<=512ull<<20&&steady_calls==16;
    out<<"{\"whole_model_loaded\":false,\"main_context_metadata_reserved_and_reset\":true,\"main_rollback_kv_clear_passed\":true,\"prefill_histogram_layers\":43,\"mtp_no_prefill_routes\":true,\"directory_layers\":44,\"mtp_layer_index\":43,\"cold_cpu_routes\":"<<cold.cpu_routes-before.cpu_routes<<",\"cached_only_cpu_routes\":"<<skipped.cpu_routes-skipped_before.cpu_routes<<",\"cached_only_token\":"<<skipped_token<<",\"cold_expert_h2d_bytes\":"<<cold.transport.bytes-before.transport.bytes<<",\"allhit_host_services\":"<<hot.host_services-hot_before.host_services<<",\"allhit_cpu_routes\":"<<hot.cpu_routes-hot_before.cpu_routes<<",\"allhit_tokens_equal\":"<<(hot_token==cached_hot_token?"true":"false")<<",\"allhit_expert_h2d_bytes\":"<<hot.transport.bytes-hot_before.transport.bytes<<",\"dense_device_bytes\":"<<dense<<",\"scratch_device_bytes\":"<<scratch<<",\"steady_input_calls\":"<<steady_calls<<",\"steady_input_ms_per_call\":"<<(steady_end.host_ms[0]-steady_begin.host_ms[0])/steady_calls<<",\"steady_input_sub_ms_per_call\":[";
    for(size_t i=0;i<8;++i){if(i)out<<',';out<<(steady_end.input_sub_ms[i]-steady_begin.input_sub_ms[i])/steady_calls;}
    out<<"],\"passed\":"<<(ok?"true":"false")<<'}';return ok;
}
}
