#pragma once
#include "native_tg_graph.hpp"
#include "native_tg_trace.hpp"
#include "device_cache.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <ostream>
namespace strata::deepseek_v4 {
inline bool native_tg_pair_probe(llama_model& model,llama_moe_cache& cache,DeviceExpertCache& service,
    ggml_backend_t backend,const float* norm,std::ostream& report) {
    if(!native_tg_cpu_only())return true;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> owner(ggml_init({1ull<<20,nullptr,true}),ggml_free);auto* c=owner.get();
    auto* x=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,2);auto* scores=ggml_new_tensor_2d(c,GGML_TYPE_F32,256,2);auto* ids=ggml_top_k(c,scores,6);auto* weights=ggml_new_tensor_3d(c,GGML_TYPE_F32,1,6,2);
    auto* lookup=ggml_get_rows(c,DeviceExpertCache::slot_map(0,2,6,cache.strata_directory(0),&service),ggml_reshape_1d(c,ggml_cont(c,ids),12));DeviceExpertCache::decorate(lookup,0,&service);
    const auto& l=model.layers[0];auto* start=build_native_tg(c,x,ids,weights,ggml_reshape_2d(c,lookup,6,2),cache.get_experts(l.ffn_gate_exps),cache.get_experts(l.ffn_up_exps),cache.get_experts(l.ffn_down_exps),l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps,model.hparams.swiglu_clamp_exp[0],0);
    auto* result=finish_native_tg(c,start,ggml_scale(c,x,.125f));ggml_set_output(result);auto* graph=ggml_new_graph_custom(c,128,false);ggml_build_forward_expand(graph,result);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(c,backend),ggml_backend_buffer_free);
    if(!buffer)throw std::runtime_error("native pair fixture allocation failed");
    Mxfp4CpuService reference(4);const Mxfp4CpuBank bank{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps,model.hparams.swiglu_clamp_exp[0]};
    std::array<float,8192> activation{},actual{},expected{};std::array<float,12> w{};w.fill(.125f);std::vector<float> contributions(12*4096);
    for(int t=0;t<2;++t)for(int i=0;i<4096;++i)activation[t*4096+i]=norm[i]*(1+t*.017f);
    ggml_backend_tensor_set(x,activation.data(),0,sizeof(activation));ggml_backend_tensor_set(weights,w.data(),0,sizeof(w));
    bool passed=true;report<<"[";
    for(int missing : {0,2,6}) {
        const auto* directory=static_cast<const int32_t*>(cache.strata_directory(0)->data);std::vector<int> hot,cold;
        for(int id=0;id<256;++id)(directory[id]>=0?hot:cold).push_back(id);
        if(hot.size()<6||cold.size()<9)throw std::runtime_error("native pair fixture residency insufficient");
        std::array<int32_t,12> route{};for(int t=0;t<2;++t)for(int r=0;r<6;++r)route[t*6+r]=r<6-missing?hot[r]:cold[r-(6-missing)+(t?std::min(missing,3):0)];
        reference.compute(bank,activation.data(),route.data(),6,contributions.data());reference.compute(bank,activation.data()+4096,route.data()+6,6,contributions.data()+6*4096);
        for(int t=0;t<2;++t)for(int i=0;i<4096;++i){float value=.125f*activation[t*4096+i];for(int r=0;r<6;++r)value+=w[t*6+r]*contributions[(t*6+r)*4096+i];expected[t*4096+i]=value;}
        std::array<float,512> route_scores{};route_scores.fill(-1000.f);
        for(int token=0;token<2;++token)for(int rank=0;rank<6;++rank)route_scores[token*256+route[token*6+rank]]=6.f-rank;
        ggml_backend_tensor_set(scores,route_scores.data(),0,sizeof(route_scores));const auto before=service.metrics();
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("native pair compute failed");service.verify();const auto after=service.metrics();ggml_backend_tensor_get(result,actual.data(),0,sizeof(actual));
        double error=0,scale=0,maximum=0;for(int i=0;i<8192;++i){const double e=double(actual[i])-expected[i];error+=e*e;scale+=double(expected[i])*expected[i];maximum=std::max(maximum,std::abs(e));}
        const auto union_missing=missing+std::min(missing,3);const auto h2d=after.transport.bytes-before.transport.bytes;
        const double rms=std::sqrt(error/std::max(scale,1e-30));const bool ok=std::isfinite(rms)&&rms<=1e-4&&maximum<=.01&&!h2d&&after.cpu_routes-before.cpu_routes==uint64_t(union_missing)&&after.dma_costs.commit_wait_ns==before.dma_costs.commit_wait_ns&&after.gpu[4]==before.gpu[4]&&(missing||after.host_services==before.host_services);
        using Read=bool(*)(ggml_backend_t,NativeRouteTrace*,bool);auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto read=reinterpret_cast<Read>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_trace_read"));NativeRouteTrace trace;
        if(read&&read(backend,&trace,true)){if(trace.epochs<2)throw std::runtime_error("pair trace positions missing");
            for(int token=0;token<2;++token){const auto& record=trace.records[(size_t(trace.epochs)-2+token)*43];
                for(int rank=0;rank<6;++rank)if(record.ids[rank]!=route[token*6+rank])throw std::runtime_error("pair trace original expert IDs differ");
                if(!trace.phase_clocks.empty()){const auto& clock=trace.phase_clocks[(size_t(trace.epochs)-2+token)*43];
                    if(!clock.publish||clock.publish>clock.join_begin||clock.join_begin>clock.cpu_ready||clock.cpu_ready>clock.join_end)throw std::runtime_error("pair phase clock order differs");}}}
        passed&=ok;if(missing)report<<',';report<<"{\"missing_per_position\":"<<missing<<",\"union_missing\":"<<union_missing<<",\"cpu_union_experts\":"<<after.cpu_routes-before.cpu_routes<<",\"expert_h2d_bytes\":"<<h2d<<",\"relative_rms\":"<<rms<<",\"max_abs\":"<<maximum<<",\"passed\":"<<(ok?"true":"false")<<'}';report.flush();
        if(!ok)throw std::runtime_error("native two-position expert gate failed");
    }
    report<<"]";return passed;
}
}
