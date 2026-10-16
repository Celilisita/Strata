#pragma once
#include "runtime.hpp"
#include "device_cache.hpp"
#include "native_tg_timing.hpp"
#include "native_tg_trace.hpp"
#include "router_prediction.hpp"
#include <filesystem>
#include <fstream>
#include <vector>
namespace strata::deepseek_v4 {
// Optional bounded diagnostics inside the normal independent generation run.
// Route writes are device-only; one payload download follows each request.
class TgDiagnostics {
    using Trace=bool (*)(ggml_backend_t,NativeRouteTrace*,bool);
    using Begin=bool (*)(ggml_backend_t);
    using Read=bool (*)(ggml_backend_t,NativeTgTimingResult*);
    ggml_backend_t backend=nullptr;Trace trace=nullptr;Begin begin=nullptr;Read read=nullptr;
    Runtime* runtime=nullptr;std::filesystem::path receipt;bool seed_saved=false;
    uint32_t epoch_start=0;std::array<uint64_t,4> weight_start{};int past=0;bool timed=false;
    std::vector<std::pair<int,NativeTgTimingResult>> samples;
public:
    explicit TgDiagnostics(Runtime& owner,const std::filesystem::path& path):runtime(&owner),receipt(path) {
        auto& runtime=owner;
        backend=device_expert_backend(runtime.context());
        if(backend){auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            trace=reinterpret_cast<Trace>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_trace_read"));
            begin=reinterpret_cast<Begin>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_timing_begin"));
            read=reinterpret_cast<Read>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_timing_read"));}
        NativeRouteTrace state;if(trace&&trace(backend,&state,false)){epoch_start=state.epochs;weight_start=state.missing_unique_weight_buckets;}
    }
    void before(int position) {past=position;timed=position>=8192&&samples.size()<32&&begin&&begin(backend);}
    void after() {
        if(!seed_saved&&past>=8192&&trace){NativeRouteTrace state;if(trace(backend,&state,false)){write_device_routing_seed(runtime->context(),receipt.string()+"-routing-seed.json");seed_saved=true;}}
        if(!timed)return;NativeTgTimingResult result;
        if(!read||!read(backend,&result))throw std::runtime_error("8K native TG timing incomplete");
        samples.emplace_back(past,result);timed=false;}
    void save(const std::filesystem::path& base) {
        std::ofstream out(base.string()+"-tg-diagnostics.json");
        out<<"{\"context_cost_samples\":[";
        for(size_t i=0;i<samples.size();++i){if(i)out<<',';const auto& s=samples[i];
            out<<"{\"past\":"<<s.first<<",\"whole_gpu_ns\":"<<s.second.whole_gpu_ns<<",\"routed_compute_ns\":"<<s.second.expert_compute_ns
               <<",\"join_including_cpu_wait_ns\":"<<s.second.join_ns<<",\"nonexpert_including_shared_ns\":"<<s.second.whole_gpu_ns-s.second.expert_ns<<'}';}
        NativeRouteTrace routes;const bool available=trace&&trace(backend,&routes,true);
        out<<"],\"timing_external_event_nodes_enabled\":"<<(samples.empty()?"false":"true")<<",\"trace_epoch_start\":"<<epoch_start
           <<",\"trace_epoch_end\":"<<routes.epochs<<",\"trace_layers\":43,\"trace_record_bytes\":32,\"trace_available\":"<<(available?"true":"false")
           <<",\"missing_unique_weight_buckets\":[";
        for(size_t i=0;i<4;++i){if(i)out<<',';out<<routes.missing_unique_weight_buckets[i]-weight_start[i];}
        out<<"]}\n";
        if(!out)throw std::runtime_error("TG diagnostics receipt failed");
        if(backend){auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            using Prediction=bool(*)(ggml_backend_t,RouterPredictionTrace*,bool);
            auto read_prediction=reinterpret_cast<Prediction>(ggml_backend_reg_get_proc_address(reg,"strata_router_prediction_trace_read"));
            RouterPredictionTrace predictions;
            if(read_prediction&&read_prediction(backend,&predictions,true)){
                if(predictions.epochs!=routes.epochs||predictions.records.size()!=routes.records.size())throw std::runtime_error("Prediction trace epoch differs");
                const size_t offset=size_t(epoch_start)*43,count=size_t(predictions.epochs-epoch_start)*43;
                std::ofstream binary(base.string()+"-router-predictions.bin",std::ios::binary);
                binary.write(reinterpret_cast<const char*>(predictions.records.data()+offset),count*sizeof(RouterPredictionRecord));
                if(!binary)throw std::runtime_error("Prediction trace receipt failed");}}
        if(available){if(routes.epochs<epoch_start)throw std::runtime_error("route trace epoch regressed");
            std::ofstream binary(base.string()+"-routes.bin",std::ios::binary);
            const size_t offset=size_t(epoch_start)*43,count=size_t(routes.epochs-epoch_start)*43;
            binary.write(reinterpret_cast<const char*>(routes.records.data()+offset),count*sizeof(NativeRouteRecord));
            if(!binary)throw std::runtime_error("TG route trace write failed");
            if(!routes.phase_clocks.empty()){
                if(routes.phase_clocks.size()!=routes.records.size())throw std::runtime_error("TG phase clock shape differs");
                std::ofstream clocks(base.string()+"-phase-clocks.bin",std::ios::binary);
                clocks.write(reinterpret_cast<const char*>(routes.phase_clocks.data()+offset),count*sizeof(NativePhaseClock));
                if(!clocks)throw std::runtime_error("TG phase clock write failed");}}
    }
};
}
