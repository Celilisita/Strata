#include "strata/qwen35/native/mtp.hpp"
#include <algorithm>
#include "strata/qwen35/native/ops.hpp"
#include "mtp_graph.hpp"
namespace strata::qwen35::native {
NativeMtp::NativeMtp(DenseModel &model,const strata::GgufFile &file,cudaStream_t stream):model_(model){
    const auto &s=model_.spec_;
    if(s.mtp.size()!=1||s.n_embd!=5120||model_.limits_.qkv_planes)
        throw std::invalid_argument("native MTP requires one Dense built-in block and unleased batch workspace");
    const auto &m=s.mtp[0];const auto &d=m.decoder;
    if(m.embedding!=s.embedding||m.output!=s.output)
        throw std::invalid_argument("dedicated MTP embedding/head inventory is not supported");
    auto matrix=[&](const strata::TensorInfo *t){
        if(!t||t->shape.size()!=2||(t->type!=12&&t->type!=14))throw std::invalid_argument("invalid built-in draft quantization");
        const size_t n=strata::qwen35::tensor_bytes(file,*t);
        auto *p=storage_.allocate(n);Matrix out{p,Quant(t->type),int(t->shape[1]),int(t->shape[0])};
        check_cuda(cudaMemcpyAsync(p,file.tensor_data(*t),n,cudaMemcpyHostToDevice,stream),"own built-in MTP payload");
        check_cuda(cudaStreamSynchronize(stream),"own MTP upload source lifetime");
        if(out.quant==Quant::Q4K){
            void *temporary=storage_.allocate(n);
            out=prepare_q4_pairs(out,temporary,stream);
            check_cuda(cudaMemcpyAsync(p,temporary,n,cudaMemcpyDeviceToDevice,stream),"own draft Q4 layout");
            check_cuda(cudaStreamSynchronize(stream),"own draft prepack lease");storage_.release(temporary);
            out.data=p;
        }
        return out;
    };
    auto parameter=[&](const strata::TensorInfo *t){
        if(!t||t->type!=0)throw std::invalid_argument("invalid draft parameter format");
        const size_t n=strata::qwen35::tensor_bytes(file,*t);auto *p=static_cast<float *>(storage_.allocate(n));
        check_cuda(cudaMemcpyAsync(p,file.tensor_data(*t),n,cudaMemcpyHostToDevice,stream),"own draft norm payload");return p;
    };
    decoder_={matrix(d.q),matrix(d.k),matrix(d.v),matrix(d.attn_out),matrix(d.ffn_gate),matrix(d.ffn_up),matrix(d.ffn_down),
        parameter(d.attn_norm),parameter(d.q_norm),parameter(d.k_norm),parameter(d.post_attn_norm)};
    projection_=matrix(m.eh_proj);enorm_=parameter(m.enorm);hnorm_=parameter(m.hnorm);head_norm_=parameter(m.output_norm);
    AttentionSchedule schedule;const int partitions[]={3,20,20,2};
    for(int i=0;i<4;++i)schedule.projections[i].partitions=partitions[i];
    schedule.rounding={32,2,capabilities(0).multiprocessors,256};
    attention_=std::make_unique<AttentionLayer>(*model_.shared_,decoder_,AttentionShape{int(s.n_embd),int(s.head_dim),
        int(s.n_head),int(s.n_head_kv),int(s.rope_dim),float(s.rms_epsilon),float(s.rope_base)},schedule);
    attention_->use_verify_split(model_.attention_work().verify_split());
    attention_->use_verify_grid(model_.attention_work().verify_grid());
    capacity_=model_.cache_capacity();const size_t row=s.n_head_kv*(s.head_dim/32)*(model_.q8_cache()?34:18);
    if(model_.kv_resident()){
        ram_=std::make_unique<RamKvCache>(model_.cache_logical_capacity(),model_.kv_resident(),int(s.n_head_kv),stream,model_.kv_selected(),model_.kv_reuse_horizon());
        key_=ram_->view().key;value_=ram_->view().value;
    }else{key_=storage_.allocate(size_t(capacity_)*row);value_=storage_.allocate(size_t(capacity_)*row);}
    auto floats=[&](size_t n){return static_cast<float *>(storage_.allocate(n*4));};
    const int h=int(s.n_embd),t=model_.max_tokens();
    embedding_=floats(size_t(t)*h);concat_=floats(size_t(t)*2*h);conditioned_=floats(size_t(t)*h);
    normed_=floats(size_t(t)*h);prefix_=floats(size_t(t)*h);seed_=floats(h);draft_seed_=floats(h);
    verify_rows_=floats(3*h);catch_rows_=floats(3*h);logits_=floats(s.n_vocab);
    device_ids_=static_cast<int *>(storage_.allocate(size_t(t)*4));picked_=static_cast<int *>(storage_.allocate(4));
    invalid_=static_cast<unsigned *>(storage_.allocate(4));greedy_bytes_=greedy_workspace(int(s.n_vocab));greedy_=storage_.allocate(greedy_bytes_);
    host_=std::make_unique<MtpPinned>();target_ids_=static_cast<int *>(storage_.allocate(40));target_flags_=reinterpret_cast<unsigned *>(target_ids_+3);
    draft_ids_=target_ids_+6;draft_flags_=reinterpret_cast<unsigned *>(target_ids_+8);
    saved_.resize(s.n_layers);
    for(size_t i=0;i<saved_.size();++i)if(s.layers[i].recurrent){
        saved_[i].state=floats(s.ssm_state*s.ssm_state*s.ssm_rank);
        saved_[i].history=floats(3*s.ssm_state*(2*s.ssm_group+s.ssm_rank));
    }
    check_cuda(cudaStreamSynchronize(stream),"own draft setup complete");
}
void NativeMtp::reserve_cache(int required,cudaStream_t stream){
    if(required<1||required>model_.logical_context())throw std::out_of_range("MTP logical KV extent");
    if(ram_){
        if(graphs_&&required>ram_->view().host_capacity)throw std::logic_error("RAM Graph reservation cannot grow");
        model_.reserve_cache(required,stream);ram_->reserve(required,stream);key_=ram_->view().key;value_=ram_->view().value;return;
    }
    const int bound=model_.kv_window()?model_.kv_window()+model_.max_tokens():model_.logical_context();
    // Reject a changed Graph lease before growing/freeing either cache owner.
    if(graphs_&&required>capacity_&&capacity_!=bound)throw std::logic_error("draft Graph reservation cannot grow");
    model_.reserve_cache(required,stream);
    if(required<=capacity_||capacity_==bound)return;
    cudaStreamCaptureStatus status{};check_cuda(cudaStreamIsCapturing(stream,&status),"draft KV capture check");
    if(status!=cudaStreamCaptureStatusNone||graphs_)throw std::logic_error("reserve draft ring before graph capture");
    check_cuda(cudaStreamSynchronize(stream),"draft KV cold growth lease");
    const int next=std::min(bound,((required+8191)/8192)*8192);
    const size_t row=model_.spec_.n_head_kv*(model_.spec_.head_dim/32)*(model_.q8_cache()?34:18);
    void *key=nullptr,*value=nullptr;
    try{key=storage_.allocate(size_t(next)*row);value=storage_.allocate(size_t(next)*row);
        check_cuda(cudaMemcpyAsync(key,key_,size_t(capacity_)*row,cudaMemcpyDeviceToDevice,stream),"grow draft K");
        check_cuda(cudaMemcpyAsync(value,value_,size_t(capacity_)*row,cudaMemcpyDeviceToDevice,stream),"grow draft V");
        check_cuda(cudaStreamSynchronize(stream),"grown draft KV payload lease");
    }catch(...){cudaStreamSynchronize(stream);if(key)storage_.release(key);if(value)storage_.release(value);throw;}
    storage_.release(key_);storage_.release(value_);key_=key;value_=value;capacity_=next;
}

}
