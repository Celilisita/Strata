#include "strata/gemma4/native/sm120_model.hpp"
#include "strata/gemma4/native/sm120_budget.hpp"
#include "strata/qwen35/spec.hpp"
#include "sm120_rope_table.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace strata::gemma4::native {
const float* Sm120Model::scalar(const TensorInfo* tensor){
    if(!tensor||tensor->type!=0)throw std::invalid_argument("Gemma F32 parameter inventory");
    const auto* host=reinterpret_cast<const float*>(file_.tensor_data(*tensor));
    for(std::size_t i=0;i<tensor->elements();++i)if(!std::isfinite(host[i]))
        throw std::invalid_argument("Gemma nonfinite norm/layer scale");
    const auto bytes=qwen35::tensor_bytes(file_,*tensor);auto* gpu=static_cast<float*>(weights_.allocate(bytes));
    primitive::check_cuda(cudaMemcpyAsync(gpu,host,bytes,cudaMemcpyHostToDevice,stream_),"Gemma F32 parameter upload");
    return gpu;
}
Sm120Model::Sm120Model(const GgufFile& file,int batch,cudaStream_t stream):
    file_(file),spec_(bind_sm120_spec(file)),stream_(stream),batch_(batch){
    if(!stream||batch<1||batch>8192)throw std::invalid_argument("Gemma SM120 stream/batch");
    // Plan proves complete body and original tied inventory before any upload.
    admit_sm120_budget(estimate_sm120_budget(file,spec_,batch).constructor_peak,"model setup");
    const auto bytes=qwen35::tensor_bytes(file,*spec_.embedding);
    auto* embedding=weights_.allocate(bytes);
    primitive::check_cuda(cudaMemcpyAsync(embedding,file.tensor_data(*spec_.embedding),bytes,cudaMemcpyHostToDevice,stream),"Gemma tied raw embedding/head upload");
    embedding_={embedding,primitive::Quant(spec_.embedding->type),262144,5376,primitive::BlockLayout::Gguf};
    final_norm_=scalar(spec_.output_norm);
    const auto* freq=spec_.rope_frequencies;
    const auto* host=reinterpret_cast<const float*>(file.tensor_data(*freq));
    for(std::size_t i=0;i<freq->elements();++i)if(!(host[i]>0))
        throw std::invalid_argument("Gemma rotary factors permit positive infinity only");
    const auto freq_bytes=qwen35::tensor_bytes(file,*freq);
    auto* gpu=static_cast<float*>(weights_.allocate(freq_bytes));frequencies_=gpu;
    primitive::check_cuda(cudaMemcpyAsync(gpu,host,freq_bytes,cudaMemcpyHostToDevice,stream),"Gemma rotary factors upload");
    projections_=std::make_unique<Nvfp4ProjectionService>(file,spec_,batch,stream);
    layers_.reserve(60);
    for(const auto& layer:spec_.layers){
        Sm120LayerWeights w{};
        w.q=projections_->matrix(*layer.q);w.k=projections_->matrix(*layer.k);
        w.v=layer.shared_raw_kv?w.k:projections_->matrix(*layer.v);w.out=projections_->matrix(*layer.out);
        w.pre=scalar(layer.attn_norm);w.post=scalar(layer.attn_post);
        w.qnorm=scalar(layer.q_norm);w.knorm=scalar(layer.k_norm);
        w.ffn={projections_->matrix(*layer.gate),projections_->matrix(*layer.up),projections_->matrix(*layer.down),
            scalar(layer.ffn_norm),scalar(layer.ffn_post),scalar(layer.scale)};
        layers_.push_back(w);
    }
    ffn_=std::make_unique<Nvfp4Ffn>(*projections_,batch,spec_.epsilon);
    auto floats=[&](std::size_t n){return static_cast<float*>(memory_.allocate(n*4));};
    auto integers=[&](std::size_t n){return static_cast<int*>(memory_.allocate(n*4));};
    const auto h=std::size_t(batch)*5376;
    hidden_[0]=floats(h);hidden_[1]=floats(h);normalized_=floats(h);
    projected_=floats(h);post_=floats(h);after_=floats(h);
    q_=floats(std::size_t(batch)*32*512);attended_=floats(std::size_t(batch)*32*512);
    k_=floats(std::size_t(batch)*16*256);v_=floats(std::size_t(batch)*16*256);
    features_=floats(5376);logits_=floats(262144);input_=integers(batch);position_=integers(1);
    invalid_=static_cast<unsigned*>(memory_.allocate(sizeof(unsigned)));
    if(pp_rope_table_requested()&&batch>3){
        rope_table_=static_cast<float*>(memory_.allocate(rope_table_bytes(batch)));
        std::cerr<<"GEMMA_PP_ROPE_TABLE enabled=1 bytes="<<rope_table_bytes(batch)
            <<" max_batch="<<batch<<" decode_old_path=1\n";
    }
    primitive::check_cuda(cudaStreamSynchronize(stream),"Gemma immutable upload source lease");
}
Sm120Model::~Sm120Model(){cudaStreamSynchronize(stream_);release_graph();}
void Sm120Model::release_graph()noexcept{
    if(executable_)cudaGraphExecDestroy(executable_);if(graph_)cudaGraphDestroy(graph_);
    executable_=nullptr;graph_=nullptr;graph_input_=nullptr;
}
void Sm120Model::release_request(){
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma request drain");
    release_graph();++epoch_;eager_decode_seen_=false;target_rows_count_=0;capacity_=0;host_position_=0;
    caches_.clear();kv_memory_.reset();attention_.reset();prompt_.clear();
}
void Sm120Model::reset(int total){
    if(total<1||total>spec_.context)throw std::invalid_argument("Gemma actual context exceeds model metadata");
    release_request();
    admit_sm120_budget(estimate_sm120_budget(file_,spec_,batch_,total).request,"request KV/attention");
    auto storage=std::make_unique<primitive::DeviceStorage>();std::vector<CacheView> caches;
    for(const auto& layer:spec_.layers){
        const int count=layer.sliding?std::min(total,1024+batch_):total;
        const auto bytes=std::size_t(count)*layer.kv_heads*(layer.head_dim/32)*34;
        caches.push_back({storage->allocate(bytes),storage->allocate(bytes),count,
            layer.kv_heads,layer.head_dim,layer.sliding?1024:0});
    }
    auto attention=std::make_unique<Sm120AttentionService>(total,batch_,stream_);
    primitive::check_cuda(cudaMemsetAsync(position_,0,sizeof(int),stream_),"Gemma device position reset");
    primitive::check_cuda(cudaMemsetAsync(invalid_,0,sizeof(unsigned),stream_),"Gemma finite state reset");
    if(target_state_)primitive::check_cuda(cudaMemsetAsync(target_state_,0,sizeof(*target_state_),stream_),"Gemma MTP target reset");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma reset initialization");
    kv_memory_=std::move(storage);caches_=std::move(caches);attention_=std::move(attention);capacity_=total;
}
std::size_t Sm120Model::resident_bytes()const{
    return weights_.bytes()+memory_.bytes()+projections_->bytes()+ffn_->bytes()+
        (attention_?attention_->bytes():0)+kv_bytes();
}
}
