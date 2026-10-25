#include "strata/runtime/capacity_storage.hpp"
#include "strata/gemma4/native/model.hpp"
#include "strata/qwen35/spec.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace strata::gemma4::native {
primitive::Matrix Model::matrix(const TensorInfo* tensor){
    if(!tensor||tensor->shape.size()!=2||(tensor->type!=12&&tensor->type!=14))
        throw std::invalid_argument("Gemma raw matrix inventory");
    const auto bytes=qwen35::tensor_bytes(file_,*tensor);
    void* gpu=weights_.allocate(bytes);
    primitive::check_cuda(cudaMemcpyAsync(gpu,file_.tensor_data(*tensor),bytes,cudaMemcpyHostToDevice,stream_),"Gemma raw matrix upload");
    return {gpu,tensor->type==12?primitive::Quant::Q4K:primitive::Quant::Q6K,
        int(tensor->shape[1]),int(tensor->shape[0]),primitive::BlockLayout::Gguf};
}
const float* Model::scalar(const TensorInfo* tensor){
    if(!tensor||tensor->type!=0)throw std::invalid_argument("Gemma F32 vector inventory");
    const auto bytes=qwen35::tensor_bytes(file_,*tensor);
    const auto* host=reinterpret_cast<const float*>(file_.tensor_data(*tensor));
    for(size_t i=0;i<tensor->elements();++i)if(!std::isfinite(host[i]))
        throw std::runtime_error("Gemma nonfinite norm or scale payload");
    auto* gpu=static_cast<float*>(weights_.allocate(bytes));
    primitive::check_cuda(cudaMemcpyAsync(gpu,host,bytes,cudaMemcpyHostToDevice,stream_),"Gemma F32 parameter upload");
    return gpu;
}
Model::Model(const GgufFile& file,int maximum_batch,cudaStream_t stream,Q6DownStorage down_storage,bool global_history):
    file_(file),spec_(bind_spec(file)),stream_(stream),maximum_batch_(maximum_batch){
    global_history_enabled_=global_history;
    if(!stream||maximum_batch<1||maximum_batch>8192)throw std::invalid_argument("Gemma stream/batch contract");
    if(down_storage!=Q6DownStorage::Gguf&&down_storage!=Q6DownStorage::Aligned224)
        throw std::invalid_argument("Gemma immutable Q6 Down inventory policy");
    q6_down_storage_=down_storage;
    embedding_=matrix(spec_.embedding);
    head_=spec_.output==spec_.embedding?embedding_:matrix(spec_.output);
    final_norm_=scalar(spec_.output_norm);
    // Positive infinity deliberately disables a rotary frequency; zero, negative,
    // or NaN factors would alter or invalidate the actual GGUF model mathematics.
    const auto* freq=spec_.rope_frequencies;
    const auto freq_bytes=qwen35::tensor_bytes(file_,*freq);
    const auto* host=reinterpret_cast<const float*>(file_.tensor_data(*freq));
    for(size_t i=0;i<freq->elements();++i)if(!(host[i]>0))throw std::runtime_error("Gemma invalid rotary factor payload");
    auto* device=static_cast<float*>(weights_.allocate(freq_bytes));frequencies_=device;
    primitive::check_cuda(cudaMemcpyAsync(device,host,freq_bytes,cudaMemcpyHostToDevice,stream_),"Gemma rotary factors upload");
    layers_.reserve(spec_.layers.size());
    for(const auto& layer:spec_.layers){
        LayerWeights w{};w.q=matrix(layer.q);w.k=matrix(layer.k);
        w.v=layer.shared_raw_kv?w.k:matrix(layer.v);w.out=matrix(layer.out);
        w.pre=scalar(layer.attn_norm);w.post=scalar(layer.attn_post);
        w.qnorm=scalar(layer.q_norm);w.knorm=scalar(layer.k_norm);
        const auto gate=matrix(layer.gate),up=matrix(layer.up);
        auto down=matrix(layer.down);
        if(q6_down_storage_==Q6DownStorage::Aligned224&&down.quant==primitive::Quant::Q6K){
            const auto begin=std::chrono::steady_clock::now();
            const size_t released=primitive::matrix_bytes(down);
            down=replace_owned_q6_down(down,weights_,stream_);
            q6_down_cold_pack_wall_s_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            ++q6_down_matrices_;q6_down_released_bytes_+=released;
            q6_down_packed_bytes_+=q6_down_weight_bytes(down);
        }
        w.ffn={gate,up,down,
            scalar(layer.ffn_norm),scalar(layer.ffn_post),scalar(layer.scale)};
        layers_.push_back(w);
    }
    weight_bytes_=weights_.bytes();
    ffn_=std::make_unique<Ffn>(layers_.front().ffn,maximum_batch,spec_.epsilon);
    auto floats=[&](size_t n){return static_cast<float*>(memory_.allocate(n*sizeof(float)));};
    auto integers=[&](size_t n){return static_cast<int*>(memory_.allocate(n*sizeof(int)));};
    const size_t h=size_t(maximum_batch)*spec_.hidden;
    hidden_[0]=floats(h);hidden_[1]=floats(h);normalized_=floats(h);
    projected_=floats(h);post_=floats(h);after_=floats(h);
    q_=floats(size_t(maximum_batch)*spec_.heads*512);
    k_=floats(size_t(maximum_batch)*16*256);v_=floats(size_t(maximum_batch)*16*256);
    attended_=floats(size_t(maximum_batch)*spec_.heads*512);
    features_=floats(spec_.hidden);logits_=floats(spec_.vocabulary);
    partials_=floats(2*((spec_.vocabulary+255)/256));
    input_=integers(maximum_batch);position_=integers(1);next_=integers(1);
    verify_features_=floats(size_t(3)*spec_.hidden);verify_logits_=floats(size_t(3)*spec_.vocabulary);
    verify_partials_=floats(size_t(3)*2*((spec_.vocabulary+255)/256));
    verify_inputs_=integers(3);verify_picks_=integers(3);draft_embedding_=floats(spec_.hidden);
    verify_packet_=static_cast<VerifyPacket*>(memory_.allocate(sizeof(VerifyPacket)));
    invalid_=&verify_packet_->invalid;
    primitive::check_cuda(cudaMemsetAsync(verify_packet_,0,sizeof(VerifyPacket),stream_),"Gemma verify packet initialization");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma immutable inventory completion");
}
void Model::reset(int maximum_total){
    if(maximum_total<1||maximum_total>spec_.context)throw std::invalid_argument("Gemma actual context reservation");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma reset drain");release_graph();
    // External graphs/borrowed cache leases must already have been destroyed by
    // their owner. Model releases only its own ordinary T1 graph here.
    ++reset_epoch_;
    cache_.clear();history_views_.clear();history_owners_.clear();kv_memory_.reset();capacity_=0;host_position_=0;
    global_verify321_enqueues_=0;global_verify336_enqueues_=0;global_verify321_request_=false;
    pp_k1024_enqueues_=0;pp_three_product333_enqueues_=0;
    global_shared_long_request_=false;
    global_shared_t2_enqueues_=0;parent_decode_attention_enqueues_=0;
    physical_global_pp_request_=false;
    physical_global_pp_enqueues_=0;parent_long_global_pp_enqueues_=0;
    // Keep the cold opt-in selection, discard the prior request/capture counts.
    compensated_physical_global_pp_request_=false;
    compensated_physical_global_pp_enqueues_=0;
    ffn_->clear_t1_q4_enqueues();ffn_->clear_t1_q6_down_enqueues();ffn_->clear_t1_gu269_enqueues();ffn_->clear_t1_gu291_enqueues();
    t1_qkv_enqueues_=0;t1_head_enqueues_=0;
    t1_single_qkv_enqueues_=0;t1_global_merge_enqueues_=0;t1_swa270_enqueues_=0;
    t1_global_shared280_enqueues_=0;t1_out302_enqueues_=0;ffn_->clear_short_pp_enqueues();
    auto storage=std::make_unique<strata::runtime::CapacityStorage>();
    std::vector<CacheView> cache;
    std::vector<std::unique_ptr<HistoryCache>> histories(spec_.layers.size());
    std::vector<HistoryView> views(spec_.layers.size());
    for(size_t index=0;index<spec_.layers.size();++index){
        const auto& layer=spec_.layers[index];
        if(global_history_enabled_&&!layer.sliding&&maximum_total>HistoryResident){
            auto owned=std::make_unique<HistoryCache>(maximum_total,HistoryResident,stream_);
            const auto view=owned->view();views[index]=view;
            // The legacy CacheView always describes its actual address space.
            // Requests that fit 32K retain the original GPU-only cache below.
            // Larger requests expose complete mapped RAM for bounded GPU PP.
            cache.push_back({view.host_key,view.host_value,
                maximum_total,layer.kv_heads,layer.head_dim,0});
            histories[index]=std::move(owned);continue;
        }
        // Append a whole PP chunk before attention. Keep enough SWA slots for
        // the earliest query's previous window as well as every new chunk row.
        const int count=layer.sliding?(std::min)(maximum_total,spec_.window+maximum_batch_):maximum_total;
        const size_t bytes=size_t(count)*layer.kv_heads*(layer.head_dim/32)*34;
        cache.push_back({storage->allocate(bytes),storage->allocate(bytes),count,
            layer.kv_heads,layer.head_dim,layer.sliding?spec_.window:0});
    }
    kv_memory_=std::move(storage);cache_=std::move(cache);
    history_owners_=std::move(histories);history_views_=std::move(views);capacity_=maximum_total;
    primitive::check_cuda(cudaMemsetAsync(position_,0,4,stream_),"Gemma position reset");
    primitive::check_cuda(cudaMemsetAsync(verify_packet_,0,sizeof(VerifyPacket),stream_),"Gemma finite/verify packet reset");
    primitive::check_cuda(cudaMemsetAsync(next_,0,4,stream_),"Gemma next ID reset");
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma reset complete");
}
}
