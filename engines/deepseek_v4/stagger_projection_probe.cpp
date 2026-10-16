#include "checkpoint.hpp"
#include "matrix_timing.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Weight {
    std::string name;
    ggml_type type;
    int64_t input, output;
    std::vector<uint8_t> data;
};

std::vector<Weight> read_attention(const char* first) {
    std::string path(first);
    const auto segment=path.find("00001-of-00004");
    if(segment==std::string::npos)throw std::runtime_error("Expected first of four GGUF shards");
    std::vector<Weight> weights;
    for(int shard=1;shard<=4;++shard) {
        path[segment+4]=char('0'+shard);
        std::unique_ptr<gguf_context,decltype(&gguf_free)> header(
            gguf_init_from_file(path.c_str(),{true,nullptr}),gguf_free);
        if(!header)throw std::runtime_error("GGUF header unavailable");
        std::ifstream input(path,std::ios::binary);
        if(!input)throw std::runtime_error("GGUF shard unavailable");
        for(int64_t id=0;id<gguf_get_n_tensors(header.get());++id) {
            const std::string name=gguf_get_tensor_name(header.get(),id);
            if(name.rfind("blk.0.attn_",0)!=0||name.find(".weight")==std::string::npos)continue;
            const auto* ne=gguf_get_tensor_ne(header.get(),id);
            const auto type=gguf_get_tensor_type(header.get(),id);
            const size_t bytes=gguf_get_tensor_size(header.get(),id);
            if(ne[0]<128||ne[1]<128||ne[2]!=1||ne[3]!=1||bytes<(1ull<<20))continue;
            if(type!=GGML_TYPE_Q8_0&&type!=GGML_TYPE_BF16&&type!=GGML_TYPE_F16&&type!=GGML_TYPE_F32)
                throw std::runtime_error("Unrecognized attention projection format: "+name);
            Weight w{name,type,ne[0],ne[1],std::vector<uint8_t>(bytes)};
            input.seekg(gguf_get_data_offset(header.get())+gguf_get_tensor_offset(header.get(),id));
            input.read(reinterpret_cast<char*>(w.data.data()),std::streamsize(bytes));
            if(!input)throw std::runtime_error("Actual attention weight payload truncated: "+name);
            weights.push_back(std::move(w));
        }
    }
    std::sort(weights.begin(),weights.end(),[](const auto& a,const auto& b){return a.name<b.name;});
    if(weights.size()<4)throw std::runtime_error("Layer-zero attention projection inventory too small");
    return weights;
}

std::vector<float> activation(int64_t input,const std::vector<float>& actual) {
    std::vector<float> x(size_t(input)*2);
    if(input==4096) {
        std::copy_n(actual.data(),x.size(),x.data());
    } else {
        for(size_t i=0;i<x.size();++i)x[i]=float(std::sin(double(i%input)*.007+double(i/input)*.13));
    }
    return x;
}

struct Graph {
    ggml_backend_t backend;
    ggml_context* meta=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_cgraph* graph=nullptr;
    std::vector<std::vector<ggml_tensor*>> outputs;
    bool split;
    Graph(ggml_backend_t b,const std::vector<Weight>& weights,const std::vector<float>& actual,bool separate)
        :backend(b),split(separate) {
        meta=ggml_init({8ull<<20,nullptr,true});
        if(!meta)throw std::runtime_error("Probe metadata unavailable");
        graph=ggml_new_graph_custom(meta,512,false);
        struct Inputs {ggml_tensor* weight;std::vector<ggml_tensor*> x;};
        std::vector<Inputs> tensors;
        for(const auto& w:weights) {
            Inputs row{};
            row.weight=ggml_new_tensor_2d(meta,w.type,w.input,w.output);
            const int count=separate?2:1;
            std::vector<ggml_tensor*> result;
            for(int p=0;p<count;++p) {
                auto* x=ggml_new_tensor_2d(meta,GGML_TYPE_F32,w.input,separate?1:2);
                auto* y=ggml_mul_mat(meta,row.weight,x);
                ggml_set_output(y);ggml_build_forward_expand(graph,y);
                row.x.push_back(x);result.push_back(y);
            }
            tensors.push_back(std::move(row));outputs.push_back(std::move(result));
        }
        buffer=ggml_backend_alloc_ctx_tensors(meta,backend);
        if(!buffer)throw std::runtime_error("Probe CUDA allocation unavailable");
        for(size_t i=0;i<weights.size();++i) {
            const auto x=activation(weights[i].input,actual);
            ggml_backend_tensor_set(tensors[i].weight,weights[i].data.data(),0,weights[i].data.size());
            for(size_t p=0;p<tensors[i].x.size();++p)
                ggml_backend_tensor_set(tensors[i].x[p],x.data()+(split?p*size_t(weights[i].input):0),0,
                    size_t(weights[i].input)*(split?1:2)*sizeof(float));
        }
    }
    ~Graph(){ggml_backend_synchronize(backend);if(buffer)ggml_backend_buffer_free(buffer);if(meta)ggml_free(meta);}
    std::vector<float> read(const std::vector<Weight>& weights) const {
        std::vector<float> all;
        for(size_t i=0;i<weights.size();++i) {
            const size_t count=size_t(weights[i].output)*(split?1:2);
            for(auto* y:outputs[i]) {
                std::vector<float> value(count);
                ggml_backend_tensor_get(y,value.data(),0,count*sizeof(float));
                all.insert(all.end(),value.begin(),value.end());
            }
        }
        return all;
    }
};

double median(std::array<double,3> x){std::sort(x.begin(),x.end());return x[1];}
}

int main(int argc,char** argv)try {
    if(argc!=4)throw std::runtime_error("Expected first GGUF shard, activation checkpoint, Y-temp JSON report");
    llama_backend_init();
    const auto checkpoint=strata::deepseek_v4::read_checkpoint_data(argv[2],"hc_attn_pre-0");
    if(checkpoint.metadata.type!=GGML_TYPE_F32||checkpoint.metadata.dimensions[0]!=4096||
       checkpoint.data.size()<8192*sizeof(float))throw std::runtime_error("Two actual attention activations unavailable");
    std::vector<float> actual(8192);
    std::memcpy(actual.data(),checkpoint.data.data(),actual.size()*sizeof(float));
    const auto weights=read_attention(argv[1]);
    auto* device=ggml_backend_dev_by_name("CUDA0");
    if(!device)throw std::runtime_error("CUDA0 unavailable");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(
        ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    if(!backend)throw std::runtime_error("CUDA backend unavailable");
    Graph paired(backend.get(),weights,actual,false),separate(backend.get(),weights,actual,true);
    const auto a1=strata::deepseek_v4::measure_matrix_graph(backend.get(),paired.graph,true);
    const auto b=strata::deepseek_v4::measure_matrix_graph(backend.get(),separate.graph,true);
    const auto a2=strata::deepseek_v4::measure_matrix_graph(backend.get(),paired.graph,true);
    const auto reference=paired.read(weights),candidate=separate.read(weights);
    if(reference.size()!=candidate.size())throw std::runtime_error("Projection result shape differs");
    double square=0,scale=0,maximum=0;
    for(size_t i=0;i<reference.size();++i) {
        if(!std::isfinite(reference[i])||!std::isfinite(candidate[i]))
            throw std::runtime_error("Nonfinite attention projection");
        const double delta=double(reference[i])-candidate[i];
        square+=delta*delta;scale+=double(reference[i])*reference[i];
        maximum=std::max(maximum,std::abs(delta));
    }
    const double relative=std::sqrt(square/std::max(scale,1e-30));
    size_t bytes=0;for(const auto& w:weights)bytes+=w.data.size();
    const double first=median(a1),second=median(a2),split=median(b);
    std::ofstream report(argv[3]);
    report<<std::setprecision(12)<<"{\"whole_model_loaded\":false,\"layer\":0,"
          <<"\"real_attention_projection_weights\":true,\"actual_4096_activation\":true,"
          <<"\"other_activation_shapes_deterministic\":true,\"l2_cold_each_replay\":true,"
          <<"\"weights\":[";
    for(size_t i=0;i<weights.size();++i) {
        if(i)report<<',';
        report<<"{\"name\":\""<<weights[i].name<<"\",\"type\":\""
              <<ggml_type_name(weights[i].type)<<"\",\"input\":"<<weights[i].input
              <<",\"output\":"<<weights[i].output<<",\"bytes\":"<<weights[i].data.size()<<'}';
    }
    report<<"],\"weight_bytes\":"<<bytes<<",\"paired_a1_ms\":"<<first
          <<",\"split_ms\":"<<split<<",\"paired_a2_ms\":"<<second
          <<",\"extra_ms_per_layer\":"<<split-(first+second)/2
          <<",\"projected_extra_ms_43_layers\":"<<(split-(first+second)/2)*43
          <<",\"relative_rms\":"<<relative<<",\"max_abs\":"<<maximum
          <<",\"numerical_passed\":"<<(relative<=1e-3?"true":"false")<<"}\n";
    if(!report||relative>1e-3)throw std::runtime_error("Projection numerical/report gate failed");
    std::cout<<"Single-layer real-weight projection A/B/A passed\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
