#include "native_tg_probe.hpp"
#include "native_tg_graph.hpp"
#include "device_cache.hpp"
#include "pp_stream_probe_graph.hpp"
#include "matrix_timing.hpp"
#include "native_tg_pressure.hpp"
#include "native_tg_pair_probe.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <memory>
#include <stdexcept>
namespace strata::deepseek_v4 {
namespace {
void env(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value)) throw std::runtime_error("native probe environment failed");
#else
    if (setenv(name, value, 1)) throw std::runtime_error("native probe environment failed");
#endif
}
bool direct_upload(ggml_backend_t b, ggml_tensor* t, const void* p, size_t offset, size_t n, void*) {
    ggml_backend_tensor_set_async(b,t,p,offset,n); return true;
}
struct ProbeGraph {
    ggml_backend_t backend;
    ggml_context* meta = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *x = nullptr, *ids = nullptr, *weights = nullptr, *out = nullptr;
    ProbeGraph(ggml_backend_t b, llama_model& model, llama_moe_cache& cache, DeviceExpertCache& service, bool native=true) : backend(b) {
        meta=ggml_init({1ull<<20,nullptr,true});
        if(!meta) throw std::runtime_error("native probe graph metadata unavailable");
        x=ggml_new_tensor_2d(meta,GGML_TYPE_F32,4096,1);
        ids=ggml_new_tensor_2d(meta,GGML_TYPE_I32,6,1);
        weights=ggml_new_tensor_3d(meta,GGML_TYPE_F32,1,6,1);
        auto* map=DeviceExpertCache::slot_map(0,1,6,cache.strata_directory(0),&service);
        auto* lookup=ggml_get_rows(meta,map,ggml_reshape_1d(meta,ids,6));
        DeviceExpertCache::decorate(lookup,0,&service);
        auto* slots=ggml_reshape_2d(meta,lookup,6,1);
        const auto& l=model.layers[0];
        if(native)out=build_native_tg(meta,x,ids,weights,slots,cache.get_experts(l.ffn_gate_exps),
                cache.get_experts(l.ffn_up_exps),cache.get_experts(l.ffn_down_exps),
                l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps,model.hparams.swiglu_clamp_exp[0],0);
        else {
            auto* act=ggml_reshape_3d(meta,x,4096,1,1);
            auto* gate=ggml_mul_mat_id(meta,cache.get_experts(l.ffn_gate_exps),act,slots);
            auto* up=ggml_mul_mat_id(meta,cache.get_experts(l.ffn_up_exps),act,slots);
            auto* h=model.hparams.swiglu_clamp_exp[0]>1e-6f?ggml_swiglu_clamp(meta,gate,up,model.hparams.swiglu_clamp_exp[0]):ggml_swiglu_split(meta,gate,up);
            auto* down=ggml_mul_mat_id(meta,cache.get_experts(l.ffn_down_exps),h,slots);
            auto* weighted=ggml_mul(meta,down,weights);
            out=ggml_view_2d(meta,weighted,4096,1,weighted->nb[2],0);
            for(int r=1;r<6;++r)out=ggml_add(meta,out,ggml_view_2d(meta,weighted,4096,1,weighted->nb[2],r*weighted->nb[1]));
            out=ggml_cont(meta,out);
        }
        if(!out) throw std::runtime_error("native probe graph seam unavailable");
        if(native_tg_cpu_only()) {
            auto* shared=ggml_scale(meta,x,.125f);
            auto* joined=native?finish_native_tg(meta,out,shared):nullptr;
            out=joined?joined:ggml_add(meta,out,shared);
        }
        ggml_set_output(out);
        graph=ggml_new_graph_custom(meta,128,false); ggml_build_forward_expand(graph,out);
        buffer=ggml_backend_alloc_ctx_tensors(meta,backend);
        if(!buffer) throw std::runtime_error("native probe graph storage unavailable");
    }
    ~ProbeGraph(){ ggml_backend_synchronize(backend); if(buffer)ggml_backend_buffer_free(buffer); if(meta)ggml_free(meta); }
    void input(const float* norm,const std::array<int32_t,6>& route,const std::vector<float>& w) {
        ggml_backend_tensor_set(x,norm,0,4096*sizeof(float));
        ggml_backend_tensor_set(ids,route.data(),0,sizeof(route));
        ggml_backend_tensor_set(weights,w.data(),0,w.size()*sizeof(float));
    }
    double compute() {
        const auto begin=std::chrono::steady_clock::now();
        if(ggml_backend_graph_compute_async(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("native probe compute failed");
        ggml_backend_synchronize(backend);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
    std::vector<float> output() { std::vector<float> v(4096); ggml_backend_tensor_get(out,v.data(),0,4096*sizeof(float)); return v; }
};
}
bool native_tg_operator_probe(const std::string& path, ggml_backend_t ref_backend,
    const std::array<ggml_tensor*,3>& source, const std::array<ggml_tensor*,3>& registered_source,
    const float* norm, float clamp, std::ostream& out) {
    env("STRATA_DSV4_NATIVE_TG","1"); env("STRATA_DSV4_DEVICE_CACHE","1");
    env("STRATA_DSV4_EXPERT_FREQUENCY","1"); env("STRATA_DSV4_CACHE_COST_TRACE","1");
    env("STRATA_DSV4_NATIVE_CPU_THREADS","4"); env("GGML_CUDA_GRAPH_OPT","0");
    const auto device=ggml_backend_get_device(ref_backend);
    ggml_backend_dev_t devices[]{device,nullptr};
    const llama_model_tensor_buft_override override_buft[]{
        {"blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight",ggml_backend_cpu_buffer_type()}, {nullptr,nullptr}};
    auto mp=llama_model_default_params(); mp.devices=devices; mp.tensor_buft_overrides=override_buft;
    mp.n_gpu_layers=-1; mp.main_gpu=0; mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    mp.use_extra_bufts=false; mp.load_mtp=false; mp.load_mode=LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode=LLAMA_LAZY_MODE_ON; mp.no_alloc=true;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(path.c_str(),mp),llama_model_free);
    if(!model||model->layers.size()!=43) throw std::runtime_error("native probe metadata model differs");
    // No dense weights or whole model inference. Distinct tensor identities
    // borrow the authentic layer-zero payload for the production 43-layer
    // cache directory contract. Only layer zero is computed in this probe.
    auto* source_meta=ggml_init({129*ggml_tensor_overhead(),nullptr,true});
    if(!source_meta) throw std::runtime_error("native probe source metadata allocation failed");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> owned_meta(source_meta,ggml_free);
    for(int il=0;il<43;++il) {
        std::array<ggml_tensor*,3> aliases{};
        for(int r=0;r<3;++r) {
            aliases[r]=ggml_new_tensor_3d(source_meta,GGML_TYPE_MXFP4,source[r]->ne[0],source[r]->ne[1],256);
            aliases[r]->data=registered_source[r]->data; aliases[r]->buffer=registered_source[r]->buffer;
            ggml_format_name(aliases[r],"probe_real_layer0.%d.%d",il,r);
        }
        model->layers[il].ffn_gate_exps=aliases[0]; model->layers[il].ffn_up_exps=aliases[1]; model->layers[il].ffn_down_exps=aliases[2];
        model->layers[il].ffn_gate_up_exps=nullptr;
    }
    model->hparams.no_alloc=false; // Allocate only the explicitly bounded cache below.
    PpProbeGraph reference(ref_backend,source,1,clamp,false,32);
    reference.upload(source); ggml_backend_synchronize(ref_backend);
    bool passed=true;
    out<<std::setprecision(12)<<"{\"scope\":\"real layer-zero math and production mixed cache service; other layer metadata aliases only\",\"dense_weights_loaded\":false,\"cases\":[";
    bool first=true;
    for(int captured=1;captured<2;++captured) {
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
        if(!backend||!cpu) throw std::runtime_error("native probe backends unavailable");
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        std::vector<ggml_backend_buffer_type_t> bufts{ggml_backend_get_default_buffer_type(backend.get()),ggml_backend_cpu_buffer_type()};
        std::unique_ptr<ggml_backend_sched,decltype(&ggml_backend_sched_free)> scheduler(
            ggml_backend_sched_new(backends.data(),bufts.data(),2,256,false,true),ggml_backend_sched_free);
        llama_moe_cache cache(*model,backends,bufts,19ull*13369344);
        cache.strata_set_leaf_aliases();
        const std::array<int32_t,6> seed{17,60,103,146,189,232};
        cache.strata_fill_selected(0,seed.data(),6,direct_upload,nullptr);
        ggml_backend_synchronize(backend.get());
        ExpertStaging staging(backend.get());
        staging.register_sources(std::vector<ggml_tensor*>(registered_source.begin(),registered_source.end()),3422552064ull);
        if(staging.source_pin_metrics().registered_bytes < 3422552064ull-6*4096)
            throw std::runtime_error("native operator source registration failed; do not benchmark artificial staging");
        DeviceExpertCache service(*model,cache,scheduler.get(),&staging);
        service.prepare(true);
        ProbeGraph native(backend.get(),*model,cache,service);
        const std::vector<float> x(norm,norm+4096), weights{.1f,.15f,.2f,.25f,.3f,.5f};
        for(int missing=0;missing<=6;++missing) {
            std::array<int32_t,6> route{};
            const auto* dir=static_cast<const int32_t*>(cache.strata_directory(0)->data);
            int used=0;
            for(int id=0;id<256&&used<6-missing;++id) if(dir[id]>=0)route[used++]=id;
            if(used!=6-missing)throw std::runtime_error("native probe resident seed unavailable");
            for(int id=0;id<256&&used<6;++id) if(dir[id]<0) {
                std::array<int32_t,6> query{}; query.fill(id); std::array<int32_t,6> result{};
                cache.strata_native_slots(0,query.data(),result.data());
                if(result[0]<0) route[used++]=id;
            }
            if(used!=6)throw std::runtime_error("native probe cold routes unavailable");
            reference.inputs(x,std::vector<int32_t>(route.begin(),route.end()),weights);
            if(ggml_backend_graph_compute_async(ref_backend,reference.graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("native reference compute failed");
            ggml_backend_synchronize(ref_backend); auto expected=reference.output();
            if(native_tg_cpu_only())for(int row=0;row<4096;++row)expected[row]+=.125f*norm[row];
            native.input(norm,route,weights);
            const auto before=service.metrics(); const double ms=native.compute(); service.verify(); const auto after=service.metrics();
            const auto actual=native.output(); double square=0,ref_square=0,max_abs=0; size_t nonfinite=0;
            for(int i=0;i<4096;++i){if(!std::isfinite(actual[i])||!std::isfinite(expected[i])){++nonfinite;continue;}
                const double e=double(actual[i])-expected[i];square+=e*e;ref_square+=double(expected[i])*expected[i];max_abs=std::max(max_abs,std::abs(e));}
            const double relative=std::sqrt(square/std::max(ref_square,1e-30));
            bool payload=true;
            std::array<int32_t,6> resolved{}; cache.strata_native_slots(0,route.data(),resolved.data());
            for(int rank=0;rank<6;++rank)if(resolved[rank]>=0)for(int r=0;r<3;++r){
                auto* bank=cache.get_experts(r==0?model->layers[0].ffn_gate_exps:r==1?model->layers[0].ffn_up_exps:model->layers[0].ffn_down_exps);
                std::vector<uint8_t> readback(bank->nb[2]);ggml_backend_tensor_get(bank,readback.data(),resolved[rank]*bank->nb[2],readback.size());
                payload&=!std::memcmp(readback.data(),static_cast<const char*>(source[r]->data)+route[rank]*source[r]->nb[2],readback.size());}
            const uint64_t cold=6*(after.gpu[0]-before.gpu[0])-(after.gpu[2]-before.gpu[2]);
            const auto cpu_routes=after.cpu_routes-before.cpu_routes;
            const auto h2d=after.transport.bytes-before.transport.bytes;
            const bool ok=!nonfinite&&relative<=1e-4&&max_abs<=.01&&payload&&cold==uint64_t(missing)&&
                !after.capacity.invalid_directory_links&&!(after.gpu[4]-before.gpu[4])&&
                (!native_tg_cpu_only()||(cpu_routes==uint64_t(missing)&&h2d==0&&after.dma_costs.commit_wait_ns==before.dma_costs.commit_wait_ns))&&
                (!missing?after.host_services==before.host_services&&cpu_routes==0&&h2d==0:after.host_services==before.host_services+1);
            passed&=ok;
            if(!first)out<<',';first=false;
            out<<"{\"capture_requested\":"<<(captured?"true":"false")<<",\"missing_routes\":"<<missing<<",\"passed\":"<<(ok?"true":"false")
               <<",\"max_abs\":"<<max_abs<<",\"relative_rms\":"<<relative<<",\"nonfinite\":"<<nonfinite<<",\"source_payload_exact\":"<<(payload?"true":"false")
               <<",\"native_ms\":"<<ms<<",\"cpu_routes\":"<<cpu_routes<<",\"expert_h2d_bytes\":"<<h2d
               <<",\"permanent_slots\":"<<after.capacity.permanent_slots<<",\"transport_slots\":"<<after.capacity.transient_slots
               <<",\"graph_preuploads\":"<<after.dma_costs.graph_preuploads-before.dma_costs.graph_preuploads<<'}';
            if(!ok)throw std::runtime_error("native real operator service gate failed: requested_missing="+std::to_string(missing)+" observed_missing="+std::to_string(cold)+" cpu="+std::to_string(cpu_routes)+" h2d="+std::to_string(h2d)+" relative="+std::to_string(relative)+" max_abs="+std::to_string(max_abs)+" payload="+std::to_string(payload)+" invalid="+std::to_string(after.capacity.invalid_directory_links));
        }
        out<<",{\"native_two_positions\":";passed&=native_tg_pair_probe(*model,cache,service,backend.get(),norm,out);out<<'}';
        // Exact zero-service replay is measured after the mixed cases, too.
        std::array<int32_t,6> route{};int used=0;
        const auto* dir=static_cast<const int32_t*>(cache.strata_directory(0)->data);
        for(int id=0;id<256&&used<6;++id)if(dir[id]>=0)route[used++]=id;
        native.input(norm,route,weights);const auto before=service.metrics();
        for(int i=0;i<3;++i)native.compute(); service.verify();const auto after=service.metrics();
        const bool replay=used==6&&after.gpu[1]-before.gpu[1]==3&&after.host_services==before.host_services&&
            after.cpu_routes==before.cpu_routes&&after.transport.bytes==before.transport.bytes&&
            after.dma_costs.frequency_d2h_bytes==before.dma_costs.frequency_d2h_bytes&&after.dma_costs.graph_preuploads>0;
        passed&=replay;
        out<<",{\"capture_requested\":"<<(captured?"true":"false")<<",\"allhit_replays\":3,\"passed\":"<<(replay?"true":"false")<<",\"host_services\":"<<after.host_services-before.host_services<<'}';
        // Matched same-input all-hit operator timing; no source uploads or CPU
        // service in either timed region. Alternate to share GPU/host conditions.
        ProbeGraph resident_reference(backend.get(),*model,cache,service,false);
        resident_reference.input(norm,route,weights);
        auto ref_compute=[&] {
            return resident_reference.compute();
        };
        for(int i=0;i<3;++i){ref_compute();native.compute();}
        std::array<double,8> a{},b{};
        const auto perf_before=service.metrics();
        for(int i=0;i<8;++i){a[i]=ref_compute();b[i]=native.compute();}
        service.verify();std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
        const double ref_ms=(a[3]+a[4])/2,native_ms=(b[3]+b[4])/2;
        auto ref_gpu=measure_matrix_graph(backend.get(),resident_reference.graph,true);
        auto native_gpu=measure_matrix_graph(backend.get(),native.graph,true);
        service.verify();std::sort(ref_gpu.begin(),ref_gpu.end());std::sort(native_gpu.begin(),native_gpu.end());
        const auto perf_after=service.metrics();
        // A whole D4 pass scans several GiB of resident expert weights, much
        // larger than L2. This gate uses the existing matched L2-cold CUDA-event
        // measurement; retain the host-wall/hot-cache result as a diagnostic.
        const bool fast=native_gpu[1]<=ref_gpu[1]*1.05&&perf_after.host_services==perf_before.host_services&&
            perf_after.cpu_routes==perf_before.cpu_routes&&perf_after.transport.bytes==perf_before.transport.bytes;passed&=fast;
        out<<",{\"allhit_reference_ms\":"<<ref_ms<<",\"allhit_native_ms\":"<<native_ms<<",\"allhit_speedup\":"<<ref_ms/native_ms
           <<",\"warm_host_wall_regression_diagnostic\":"<<native_ms/ref_ms-1
           <<",\"l2_cold_reference_gpu_ms\":"<<ref_gpu[1]<<",\"l2_cold_native_gpu_ms\":"<<native_gpu[1]
           <<",\"l2_cold_gpu_speedup\":"<<ref_gpu[1]/native_gpu[1]
           <<",\"reference_includes_device_directory_and_frequency\":true,\"shared_backend_weights_and_worker_conditions\":true"
           <<",\"l2_cold_no_more_than5percent_regression\":"<<(fast?"true":"false")<<",\"source_registered_bytes\":"<<staging.source_pin_metrics().registered_bytes<<'}';
    }
    out<<",{\"native_pressure\":";
    passed&=native_tg_pressure_probe(*model,device,registered_source,norm,out);
    out<<'}';
    env("GGML_CUDA_DISABLE_GRAPHS","1"); env("STRATA_DSV4_NATIVE_TG","0");
    out<<"],\"passed\":"<<(passed?"true":"false")<<",\"end_to_end_target_met\":false}";
    return passed;
}
}
