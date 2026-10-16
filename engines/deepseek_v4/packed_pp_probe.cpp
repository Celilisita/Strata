#include "packed_expert_store.hpp"
#include "pp_transport.hpp"
#include "pp_transport_probe.hpp"
#include "pp_cache_probe.hpp"
#include "pp_stream_probe_graph.hpp"
#include "benchmark_fault_guard.hpp"
#include "checkpoint.hpp"
#include "packed_prefetch_probe.hpp"
#include "llama-model.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
using namespace strata::deepseek_v4;
struct PackedBinding {
    ggml_backend_t backend;PackedExpertStore& store;const ggml_tensor* original;
    PackedBinding(ggml_backend_t b,PackedExpertStore& s,const ggml_tensor* key):backend(b),store(s),original(key){
        bind_pp_packed(backend,[](const ggml_tensor* gate,int first,int count,void* owner){
            auto& self=*static_cast<PackedBinding*>(owner);
            // The pressure fixture aliases authenticated layer0 payload under
            // distinct layer keys; production binding requires exact keys.
            if(!gate||gate->nb[2]!=self.original->nb[2]||gate->data!=self.original->data)
                throw std::runtime_error("packed fixture key/payload identity differs");
            return self.store.span(self.original,first,count);
        },this);
    }
    ~PackedBinding(){bind_pp_packed(backend,nullptr,nullptr);}
};
double compute(PpProbeGraph& graph){const auto start=std::chrono::steady_clock::now();
    if(ggml_backend_graph_compute_async(graph.backend,graph.graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("packed PP compute failed");
    ggml_backend_synchronize(graph.backend);return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
bool compare(const std::vector<float>& expected,const std::vector<float>& actual,double& relative,double& maximum){
    if(expected.size()!=actual.size())throw std::runtime_error("packed PP output dimensions differ");
    double error=0,reference=0;maximum=0;
    for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i])||!std::isfinite(expected[i]))return false;
        const double delta=double(actual[i])-expected[i];error+=delta*delta;reference+=double(expected[i])*expected[i];maximum=std::max(maximum,std::abs(delta));}
    relative=std::sqrt(error/std::max(reference,1e-30));return relative<=1e-4&&maximum<=.01;
}
int run(int argc,char** argv){
    if(argc!=4)throw std::runtime_error("Expected model, layer checkpoint directory and new report path");
    llama_backend_init();ggml_backend_dev_t devices[]{nullptr};auto params=llama_model_default_params();
    params.devices=devices;params.n_gpu_layers=0;params.load_mode=LLAMA_LOAD_MODE_MMAP;params.lazy_mode=LLAMA_LAZY_MODE_ON;
    params.use_extra_bufts=false;params.load_mtp=false;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],params),llama_model_free);
    if(!model||model->arch!=LLM_ARCH_DEEPSEEK4)throw std::runtime_error("packed PP model metadata unavailable");
    auto* device=ggml_backend_dev_by_name("CUDA0");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    if(!backend)throw std::runtime_error("packed PP CUDA backend unavailable");
    auto& layer=model->layers[0];const std::array<ggml_tensor*,3> sources{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps};
    const std::vector<ggml_tensor*> inventory(sources.begin(),sources.end());
    PackedExpertStore packed(backend.get(),inventory);
    ExpertSourcePin original_pin(backend.get(),inventory,3422552064ull);
    PpTransportProbeBinding ranges(backend.get(),original_pin,sources);
    const auto activation=read_checkpoint_data(argv[2],"ffn_norm-0");
    if(activation.data.size()!=4096*5*sizeof(float))throw std::runtime_error("packed PP real activation dimensions differ");
    std::vector<float> x(size_t(8192)*4096),weights(8192*6,.25f);std::vector<int32_t> ids(8192*6);
    for(int token=0;token<8192;++token){
        std::memcpy(x.data()+size_t(token)*4096,activation.data.data()+size_t(token%5)*4096*sizeof(float),4096*sizeof(float));
        for(int rank=0;rank<6;++rank)ids[token*6+rank]=(token*17+rank*43)%256;
    }
    std::vector<float> expected;
    {PpProbeGraph canonical(backend.get(),sources,8192,model->hparams.swiglu_clamp_exp[0],false,32);
        canonical.inputs(x,ids,weights);canonical.upload(sources);compute(canonical);expected=canonical.output();}
    PpProbeGraph candidate(backend.get(),sources,8192,model->hparams.swiglu_clamp_exp[0],true,32);candidate.inputs(x,ids,weights);
    std::array<double,3> control_ms{},packed_ms{};
    compute(candidate);
    for(int i=0;i<3;++i)control_ms[i]=compute(candidate);
    const auto control_output=candidate.output();
    double maximum=0,relative=0;bool passed=compare(expected,control_output,relative,maximum);
    std::ostringstream cache_report;
    uint64_t payload=0,staged=0,calls=0;
    uint64_t prefetched=0,prefetch_consumed=0;bool prefetch_passed=false;
    {
        PackedBinding binding(backend.get(),packed,sources[0]);compute(candidate);
        const auto start_metrics=pp_stream_backend_metrics(backend.get());
        for(int i=0;i<3;++i)packed_ms[i]=compute(candidate);
        const auto finish_metrics=pp_stream_backend_metrics(backend.get());
        payload=finish_metrics.payload_bytes-start_metrics.payload_bytes;staged=finish_metrics.staged_bytes-start_metrics.staged_bytes;
        calls=finish_metrics.dma_calls-start_metrics.dma_calls;
        passed&=compare(expected,candidate.output(),relative,maximum)&&payload==3*3422552064ull&&staged==0;
        prefetch_passed=packed_prefetch_probe(candidate,sources,x.data(),prefetched,prefetch_consumed)&&
            compare(expected,candidate.output(),relative,maximum);passed&=prefetch_passed;
        passed&=pp_cache_operator_probe(argv[1],backend.get(),sources,x,ids,weights,expected,model->hparams.swiglu_clamp_exp[0],32,cache_report);
    }
    Mxfp4CpuService cpu(4);const int32_t expert=17;std::vector<float> original_cpu(4096),packed_cpu(4096);
    cpu.compute({sources[0],sources[1],sources[2],model->hparams.swiglu_clamp_exp[0]},x.data(),&expert,1,original_cpu.data());
    cpu.compute(packed.cpu_bank(sources[0],model->hparams.swiglu_clamp_exp[0]),x.data(),&expert,1,packed_cpu.data());
    double cpu_relative=0,cpu_maximum=0;const bool cpu_passed=compare(original_cpu,packed_cpu,cpu_relative,cpu_maximum);passed&=cpu_passed;
    std::sort(control_ms.begin(),control_ms.end());std::sort(packed_ms.begin(),packed_ms.end());
    std::ofstream report(argv[3]);report<<std::setprecision(12)<<"{\"scope\":\"real layer0 only;8192 actual activation rows;no dense or full expert model load\","
        <<"\"status\":\""<<(passed?"passed":"validation_failure")<<"\",\"packed_source_bytes\":"<<packed.metrics().registered_bytes
        <<",\"triplet_bytes\":13369344,\"payload_exact_all_experts\":true,\"control_median_ms\":"<<control_ms[1]<<",\"packed_median_ms\":"<<packed_ms[1]
        <<",\"three_samples_h2d_bytes\":"<<payload<<",\"three_samples_dma_calls\":"<<calls<<",\"staged_bytes\":"<<staged
        <<",\"relative_rms\":"<<relative<<",\"max_abs\":"<<maximum<<",\"cpu_packed_view_passed\":"<<(cpu_passed?"true":"false")
        <<",\"prefetch_passed\":"<<(prefetch_passed?"true":"false")<<",\"prefetched_bytes\":"<<prefetched<<",\"prefetch_consumed_bytes\":"<<prefetch_consumed<<",\"cpu_max_abs\":"<<cpu_maximum<<",\"cache_reuse\":"<<cache_report.str()<<",\"end_to_end_target_met\":false}\n";
    if(!report)throw std::runtime_error("packed PP report write failed");return passed?0:2;
}
int main(int argc,char** argv){benchmark_fault_guard();try{return run(argc,argv);}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
