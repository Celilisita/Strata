#include "hybrid_provider.hpp"
#include "mtp_options.hpp"
#include "cpu_service_options.hpp"
#include <chrono>
#include "request.hpp"
#include "llama-model.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace strata::glm5_next {
bool hybrid_requested(){
    const char* value=std::getenv("STRATA_GLM53_HYBRID");
    if(value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))throw std::runtime_error("GLM hybrid flag must be0 or1");
    return value&&!std::strcmp(value,"1");
}
HybridProvider::HybridProvider(const llama_model& model,const std::string& path,const char* gpu_description,bool four_positions):model_(model),threads_(cpu_service_threads()),service_(model,threads_),four_positions_(four_positions),curves_(model.layers.size()),pair_curves_(model.layers.size()){
    if(model.arch!=LLM_ARCH_GLM5_NEXT||path.empty())throw std::runtime_error("GLM hybrid needs measured model-specific curves");
    const char* force=std::getenv("GGML_CUDA_FORCE_MMQ");
    if(force&&std::atoi(force))throw std::runtime_error("hybrid negative slots require the validated single-token MMVQ path");
    std::ifstream file(path);if(!file)throw std::runtime_error("provider curve file unavailable");
    const auto record=Json::parse(file);
    if(!record.value("passed_dequantized_reference",false)||!record.value("passed_bridge",false))throw std::runtime_error("provider curves lack same-input dequantized-reference validation");
    if(record.at("model_name")!=model.gguf_kv.at("general.name")||record.at("gpu_description")!=gpu_description)
        throw std::runtime_error("provider curves belong to a different model or GPU");
    for(size_t layer=3;layer<model.layers.size();++layer){
        const auto& row=model.layers[layer];if(!row.ffn_up_exps)continue;auto& curve=curves_[layer];
        for(const auto& sample:record.at("samples")){
            if(sample.value("cpu_threads",0)!=threads_||sample.at("gate_type")!=ggml_type_name(row.ffn_gate_exps->type)||sample.at("up_type")!=ggml_type_name(row.ffn_up_exps->type)||sample.at("down_type")!=ggml_type_name(row.ffn_down_exps->type))continue;
            const int count=sample.at("experts");const int i=count==1?0:count==4?1:count==8?2:-1;if(i<0)continue;
            if(uint64_t(sample.at("payload_bytes"))!=uint64_t(count)*(row.ffn_gate_exps->nb[2]+row.ffn_up_exps->nb[2]+row.ffn_down_exps->nb[2]))throw std::runtime_error("provider measured payload layout differs");
            if(!sample.value("passed_dequantized_reference",false))throw std::runtime_error("provider sample is numerically unvalidated");
            curve.cpu[i]=sample.at("cpu_ms");curve.cpu_control[i]=sample.value("cpu_control_ms",curve.cpu[i]);curve.wide=sample.value("cpu_variant",std::string())=="iq3_wide";curve.gpu[i]=double(sample.at("gpu_compute_ms"))+double(sample.at("pinned_staging_h2d_ms"));
        }
        curve.ready=true;
        for(int i=0;i<3;++i)if(!std::isfinite(curve.cpu[i])||!std::isfinite(curve.gpu[i])||curve.cpu[i]<=0||curve.gpu[i]<=0)throw std::runtime_error("provider cost curve is incomplete");
    }
    gu_q8_preparation_.runtime_requested=service_.gu_q8_reuse_runtime_requested();
    const auto single_prepare_begin=std::chrono::steady_clock::now();
    service_.prepare(); // graph/thread resources outside real-miss/all-hit boundaries
    gu_q8_preparation_.single_graph_prepare_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-single_prepare_begin).count();
    if(mtp_requested()) {
        if(!record.value("passed_pair",false))throw std::runtime_error("two-position provider curves lack bridge validation");
        for(size_t layer=3;layer<model.layers.size();++layer) {
            const auto& row=model.layers[layer];if(!row.ffn_up_exps)continue;
            for(const auto& sample:record.at("pair_samples")) {
                if(sample.at("cpu_threads")!=threads_||sample.at("gate_type")!=ggml_type_name(row.ffn_gate_exps->type)||
                   sample.at("up_type")!=ggml_type_name(row.ffn_up_exps->type)||sample.at("down_type")!=ggml_type_name(row.ffn_down_exps->type))continue;
                PairCurve curve;curve.count=sample.at("missing_routes");curve.unique=sample.at("unique_missing");
                curve.wide=sample.value("cpu_variant",std::string())=="iq3_wide";curve.cpu=sample.at("cpu_ms");curve.cpu_control=sample.value("cpu_control_ms",curve.cpu);curve.gpu=sample.at("gpu_compute_ms");curve.upload=sample.at("pinned_staging_h2d_ms");
                if(curve.count<1||curve.count>16||curve.unique<1||curve.unique>curve.count||
                   !std::isfinite(curve.cpu)||!std::isfinite(curve.gpu)||!std::isfinite(curve.upload)||
                   curve.cpu<=0||curve.gpu<=0||curve.upload<=0)throw std::runtime_error("pair provider curve invalid");
                pair_curves_[layer].push_back(curve);
            }
            if(pair_curves_[layer].empty())throw std::runtime_error("pair provider format curve absent");
        }
        const auto routed_prepare_begin=std::chrono::steady_clock::now();
        service_.prepare_routed_rows();
        gu_q8_preparation_.routed_graph_prepare_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-routed_prepare_begin).count();
    }
    if(four_positions_) {
        four_before_=service_.routed_rows_capacity();const auto start=std::chrono::steady_clock::now();
        service_.prepare_routed_rows_four();
        four_prepare_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        four_after_=service_.routed_rows_capacity();
    }
    // Capture capacity once while this provider has not published any jobs.
    const auto routed_capacity=service_.routed_rows_capacity();
    gu_q8_preparation_.prepared_typed_layers=routed_capacity.gu_q8_prepared_layers;
    gu_q8_preparation_.prepared_typed_graphs=routed_capacity.gu_q8_prepared_graphs;
    gu_q8_preparation_.prepared_stock_routed_graphs=routed_capacity.prepared_graphs-routed_capacity.gu_q8_prepared_graphs;
    gu_q8_preparation_.prepared_routed_graphs=routed_capacity.prepared_graphs;
    gu_q8_preparation_.routed_scheduler_cpu_buffer_bytes=routed_capacity.scheduler_cpu_buffer_bytes;
    gu_q8_preparation_.routed_activation_capacity_bytes=routed_capacity.activation_capacity_bytes;
}
bool HybridProvider::prefer_cpu(int layer,int count)const{
    if(count<1||count>8||layer<3||size_t(layer)>=curves_.size())throw std::runtime_error("provider missing set differs");
    const char* probe=std::getenv("STRATA_GLM53_STATE_PROBE_CPU");if(probe&&!std::strcmp(probe,"1"))return true;
    const auto& curve=curves_[layer];
    const int high=count<=4?1:2,low=high-1;const int lo=high==1?1:4,hi=high==1?4:8;
    const double mix=double(count-lo)/(hi-lo);
    const auto& costs=curve.wide&&!cpu_wide_requested()?curve.cpu_control:curve.cpu;
    const double cpu=costs[low]*(1-mix)+costs[high]*mix;
    const double gpu=curve.gpu[low]*(1-mix)+curve.gpu[high]*mix;
    // Include a conservative50us control/contribution allowance; end-to-end
    // counters measure the actual bridge and can invalidate this choice.
    return cpu+0.05<gpu;
}
bool HybridProvider::cpu_only_layer(int layer)const {
    for(int missing=1;missing<=8;++missing)if(!prefer_cpu(layer,missing))return false;
    if(!mtp_requested())return true;
    for(int missing=1;missing<=16;++missing)
        for(int unique=(missing+1)/2;unique<=missing;++unique)
            if(!prefer_cpu_pair(layer,missing,unique))return false;
    return true;
}
uint64_t HybridProvider::compute(int layer,const int32_t* ids,int count,const float* activation,float* output){
    service_.compute(layer,activation,ids,size_t(count),output);
    const auto& row=model_.layers.at(layer);
    return uint64_t(count)*(row.ffn_gate_exps->nb[2]+row.ffn_up_exps->nb[2]+row.ffn_down_exps->nb[2]);
}
bool HybridProvider::prefer_cpu_pair(int layer,int missing,int unique)const {
    if(layer<3||size_t(layer)>=pair_curves_.size()||missing<1||missing>16||unique<1||unique>missing||pair_curves_[layer].empty())
        throw std::runtime_error("pair provider request outside measured layouts");
    const char* probe=std::getenv("STRATA_GLM53_STATE_PROBE_CPU");if(probe&&!std::strcmp(probe,"1"))return true;
    const PairCurve* nearest=nullptr;int distance=1000;
    for(const auto& curve:pair_curves_[layer]) {
        const int d=std::abs(curve.count-missing)+2*std::abs(curve.unique-unique);
        if(d<distance){distance=d;nearest=&curve;}
    }
    // Interpolate actual missing-route CPU work and unique payload transfer.
    // GPU timing is the full canonical 8x2 shape measured by the same probe.
    const char* schedule=std::getenv(nearest->wide?"STRATA_GLM53_CPU_IQ3_WIDE":"STRATA_GLM53_CPU_PAIR_DOT");
    const double measured_cpu=schedule&&!std::strcmp(schedule,"1")?nearest->cpu:nearest->cpu_control;
    const double cpu=measured_cpu*double(missing)/nearest->count;
    const double gpu=nearest->gpu+nearest->upload*double(unique)/nearest->unique;
    return cpu+.05<gpu;
}
uint64_t HybridProvider::compute_pair(int layer,const int32_t* ids,const uint8_t* positions,int count,const float* activation,float* output) {
    int unique=0;for(int i=0;i<count;++i){bool seen=false;for(int j=0;j<i;++j)seen|=ids[j]==ids[i];if(!seen)++unique;}
    const auto start=std::chrono::steady_clock::now();
    service_.compute_routed_rows(layer,activation,ids,positions,size_t(count),output);
    const size_t bin=size_t(count)*17+unique;++pair_jobs_[bin];
    pair_ns_[bin]+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());
    const auto& row=model_.layers.at(layer);
    return uint64_t(count)*(row.ffn_gate_exps->nb[2]+row.ffn_up_exps->nb[2]+row.ffn_down_exps->nb[2]);
}

uint64_t HybridProvider::compute_four(int layer,const int32_t* ids,const uint8_t* positions,int count,const float* activation,float* output) {
    if(!four_positions_||layer<3||size_t(layer)>=model_.layers.size()||!ids||!positions||!activation||!output||count<1||count>32)
        throw std::runtime_error("four-position provider requires explicit prepared cold-CPU diagnostic");
    int unique=0;for(int i=0;i<count;++i){bool seen=false;for(int j=0;j<i;++j)seen|=ids[j]==ids[i];if(!seen)++unique;}
    const auto start=std::chrono::steady_clock::now();
    service_.compute_routed_rows_four(layer,activation,ids,positions,size_t(count),output);
    const size_t bin=size_t(count)*33+unique;++four_jobs_[bin];
    four_ns_[bin]+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());
    const auto& row=model_.layers.at(layer);
    return uint64_t(count)*(row.ffn_gate_exps->nb[2]+row.ffn_up_exps->nb[2]+row.ffn_down_exps->nb[2]);
}

}
