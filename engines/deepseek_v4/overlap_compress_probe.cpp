#include "overlap_compress.hpp"
#include "matrix_timing.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include <algorithm>
#include <array>
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
using namespace strata::deepseek_v4;
std::vector<float> read_norm(const char * path, const char * name) {
    std::string shard_path(path);
    const auto segment = shard_path.find("00001-of-00004");
    if (segment == std::string::npos) throw std::runtime_error("Expected first of four GGUF shards");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(nullptr, gguf_free);
    int64_t id = -1;
    for (int shard = 1; shard <= 4; ++shard) {
        shard_path[segment + 4] = char('0' + shard);
        file.reset(gguf_init_from_file(shard_path.c_str(), {true, nullptr}));
        if (!file) throw std::runtime_error("Actual GGUF shard header unavailable: " + shard_path);
        id = gguf_find_tensor(file.get(), name);
        if (id >= 0) break;
    }
    if (id < 0) throw std::runtime_error(std::string("Compressor norm missing in all four GGUF shards: ") + name);
    if (gguf_get_tensor_type(file.get(), id) != GGML_TYPE_F32 ||
        gguf_get_tensor_ne(file.get(), id)[1] != 1 || gguf_get_tensor_size(file.get(), id) > 512 * sizeof(float))
        throw std::runtime_error(std::string("Compressor norm has unexpected type or dimensions: ") + name +
            "; type=" + ggml_type_name(gguf_get_tensor_type(file.get(), id)) +
            "; size=" + std::to_string(gguf_get_tensor_size(file.get(), id)) +
            "; ne0=" + std::to_string(gguf_get_tensor_ne(file.get(), id)[0]) +
            "; ne1=" + std::to_string(gguf_get_tensor_ne(file.get(), id)[1]));
    const auto size = gguf_get_tensor_size(file.get(), id);
    std::vector<float> norm(size / sizeof(float));
    std::ifstream input(shard_path, std::ios::binary);
    input.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), id));
    input.read(reinterpret_cast<char *>(norm.data()), size);
    if (!input || std::any_of(norm.begin(), norm.end(), [](float x) { return !std::isfinite(x); }))
        throw std::runtime_error("Actual compressor norm payload differs");
    return norm;
}
struct Data {
    int dim, rows=19, blocks=2, ratio=4;
    std::vector<float> kv, score, norm;
    std::vector<int32_t> indices;
    explicit Data(std::vector<float> weight) : dim(int(weight.size())), norm(std::move(weight)) {
        if (dim < 1 || dim > 512) throw std::runtime_error("Compressor dimension unsupported");
        kv.resize(2 * dim * rows); score.resize(kv.size()); indices.resize(2 * ratio * blocks);
        for (size_t i=0;i<kv.size();++i) {
            kv[i] = .3f * std::sin(float(i) * .007f) + .04f * float(int(i % 11)-5);
            score[i] = .7f * std::cos(float(i) * .011f) + .02f * float(int(i % 13)-6);
        }
        for (size_t i=0;i<indices.size();++i) indices[i]=i%7==0?rows:int((i*5+3)%rows);
    }
};
struct Graph {
    ggml_backend_t backend;
    ggml_context * context=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_cgraph * graph=nullptr;
    ggml_tensor * result=nullptr;
    Graph(ggml_backend_t b,const Data& data,float eps,bool fused):backend(b) {
        context=ggml_init({1ull<<20,nullptr,true});
        if(!context)throw std::runtime_error("Compressor graph metadata allocation failed");
        auto* kv=ggml_new_tensor_2d(context,GGML_TYPE_F32,2*data.dim,data.rows);
        auto* score=ggml_new_tensor_2d(context,GGML_TYPE_F32,2*data.dim,data.rows);
        auto* indices=ggml_new_tensor_1d(context,GGML_TYPE_I32,int(data.indices.size()));
        auto* norm=ggml_new_tensor_1d(context,GGML_TYPE_F32,data.dim);
        if(fused)result=build_overlap_compress(context,kv,score,indices,norm,data.ratio,data.blocks,eps);
        else {
            auto* zero_kv=ggml_scale(context,ggml_view_1d(context,kv,2*data.dim,0),0.f);
            auto* zero_score=ggml_scale_bias(context,ggml_view_1d(context,score,2*data.dim,0),0.f,-INFINITY);
            auto* kv_aug=ggml_concat(context,kv,ggml_reshape_2d(context,zero_kv,2*data.dim,1),1);
            auto* score_aug=ggml_concat(context,score,ggml_reshape_2d(context,zero_score,2*data.dim,1),1);
            auto* kv_rows=ggml_get_rows(context,kv_aug,indices);
            auto* score_rows=ggml_get_rows(context,score_aug,indices);
            const int n_read=data.ratio*data.blocks;
            auto half=[&](ggml_tensor* x,bool current) {
                const size_t offset=current?size_t(n_read)*x->nb[1]+size_t(data.dim)*sizeof(float):0;
                auto* view=ggml_view_2d(context,x,data.dim,n_read,x->nb[1],offset);
                return ggml_reshape_3d(context,ggml_cont(context,view),data.dim,data.ratio,data.blocks);
            };
            auto* values=ggml_concat(context,half(kv_rows,false),half(kv_rows,true),1);
            auto* scores=ggml_concat(context,half(score_rows,false),half(score_rows,true),1);
            values=ggml_cont(context,ggml_permute(context,values,1,0,2,3));
            scores=ggml_cont(context,ggml_permute(context,scores,1,0,2,3));
            auto* comp=ggml_sum_rows(context,ggml_mul(context,values,ggml_soft_max(context,scores)));
            comp=ggml_cont(context,ggml_permute(context,comp,1,0,2,3));
            result=ggml_mul(context,ggml_rms_norm(context,comp,eps),norm);
        }
        ggml_set_output(result);
        graph=ggml_new_graph_custom(context,256,false);ggml_build_forward_expand(graph,result);
        buffer=ggml_backend_alloc_ctx_tensors(context,backend);
        if(!buffer)throw std::runtime_error("Compressor graph GPU allocation failed");
        ggml_backend_tensor_set(kv,data.kv.data(),0,data.kv.size()*sizeof(float));
        ggml_backend_tensor_set(score,data.score.data(),0,data.score.size()*sizeof(float));
        ggml_backend_tensor_set(indices,data.indices.data(),0,data.indices.size()*sizeof(int32_t));
        ggml_backend_tensor_set(norm,data.norm.data(),0,data.norm.size()*sizeof(float));
    }
    ~Graph(){if(backend)ggml_backend_synchronize(backend);if(buffer)ggml_backend_buffer_free(buffer);if(context)ggml_free(context);}
    std::vector<float> read(int dim,int blocks){std::vector<float> x(size_t(dim)*blocks);ggml_backend_tensor_get(result,x.data(),0,x.size()*sizeof(float));return x;}
};
double median(std::array<double,3> x){std::sort(x.begin(),x.end());return x[1];}
void run_case(ggml_backend_t backend,const char* model,const char* name,std::ostream& out,bool comma){
    Data data(read_norm(model,name));constexpr float eps=1e-6f;
    Graph control(backend,data,eps,false),candidate(backend,data,eps,true);
    const auto before=measure_matrix_graph(backend,control.graph);
    const auto a=control.read(data.dim,data.blocks);
    const auto after=measure_matrix_graph(backend,candidate.graph);
    const auto b=candidate.read(data.dim,data.blocks);
    double squared=0,base=0,maximum=0;for(size_t i=0;i<a.size();++i){const double diff=double(a[i])-b[i];squared+=diff*diff;base+=double(a[i])*a[i];maximum=std::max(maximum,std::abs(diff));}
    const double relative=std::sqrt(squared/std::max(base,1e-30));
    if(!std::isfinite(relative)||relative>2e-4||maximum>2e-3)
        throw std::runtime_error("Fused compressor real-weight numerical gate failed: rel=" +
            std::to_string(relative) + "; max=" + std::to_string(maximum) +
            "; first control=" + std::to_string(a[0]) + "; first fused=" + std::to_string(b[0]));
    if(comma)out<<',';out<<std::setprecision(10)<<"{\"tensor\":\""<<name<<"\",\"dimension\":"<<data.dim
        <<",\"blocks\":"<<data.blocks<<",\"control_gpu_ms\":"<<median(before)
        <<",\"fused_gpu_ms\":"<<median(after)<<",\"relative_rms\":"<<relative
        <<",\"max_abs\":"<<maximum<<",\"numerical_passed\":true}";
}
}
int main(int argc,char** argv)try {
    if(argc!=3)throw std::runtime_error("Expected main GGUF shard and new Y-temp report path");
    llama_backend_init();auto* device=ggml_backend_dev_by_name("CUDA0");if(!device)throw std::runtime_error("CUDA0 missing");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    std::ofstream out(argv[2]);out<<"{\"whole_model_loaded\":false,\"actual_norm_weights\":true,\"cases\":[";
    run_case(backend.get(),argv[1],"blk.2.attn_compressor_norm.weight",out,false);
    run_case(backend.get(),argv[1],"blk.2.indexer_compressor_norm.weight",out,true);
    out<<"],\"passed\":true}\n";if(!out)throw std::runtime_error("Compressor report write failed");
    std::cout<<"passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
