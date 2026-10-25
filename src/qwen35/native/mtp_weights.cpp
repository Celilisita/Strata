#include "device_selected_request.hpp"
#include "resident_kv_policy.hpp"
#include "t1_tc.hpp"
#include "strata/qwen35/native/mtp.hpp"
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include "strata/qwen35/native/ops.hpp"
#include "mtp_graph.hpp"
namespace strata::qwen35::native {
NativeMtp::NativeMtp(DenseModel &model,const strata::GgufFile &file,cudaStream_t stream,int max_proposals):model_(model),proposal_capacity_(max_proposals){
    if(max_proposals<1||max_proposals>3)throw std::invalid_argument("MTP proposal capacity");
    if(max_proposals==3){const auto caps=capabilities(0);
        if(caps.major!=12||caps.minor!=0)throw std::invalid_argument("MTP3 requires measured SM120");}
    const auto &s=model_.spec_;
    if(s.mtp.size()!=1||s.n_embd!=5120||model_.limits_.qkv_planes)
        throw std::invalid_argument("native MTP requires one Dense built-in block and unleased batch workspace");
    const auto &m=s.mtp[0];const auto &d=m.decoder;
    if(m.embedding!=s.embedding||m.output!=s.output)
        throw std::invalid_argument("dedicated MTP embedding/head inventory is not supported");
    auto matrix=[&](const strata::TensorInfo *t){
        if(t&&t->type==40)return model_.weights_.borrowed_mtp_nvfp4(*t,stream);
        if(!t||t->shape.size()!=2||(t->type!=12&&t->type!=14))
            throw std::invalid_argument("draft matrix requires registered NVFP4 or legacy Q4_K/Q6_K; raw Q8 is unavailable");
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
    attention_->use_draft_t1_tc(max_proposals==2&&t1_tc_enabled());
    attention_->use_verify_split(model_.attention_work().verify_split());
    attention_->use_verify_grid(model_.attention_work().verify_grid());
    capacity_=model_.cache_capacity();const size_t row=s.n_head_kv*(s.head_dim/32)*(model_.q8_cache()?34:18);
    if(model_.kv_resident()){
        ram_=std::make_unique<RamKvCache>(model_.cache_logical_capacity(),model_.kv_resident(),int(s.n_head_kv),stream,model_.kv_selected(),model_.kv_reuse_horizon(),model_.ram_bytes()==0&&model_.cache_logical_capacity()<=model_.kv_resident());
        key_=ram_->view().key;value_=ram_->view().value;
    }else{key_=storage_.allocate(size_t(capacity_)*row);value_=storage_.allocate(size_t(capacity_)*row);}
    auto floats=[&](size_t n){return static_cast<float *>(storage_.allocate(n*4));};
    const int h=int(s.n_embd),t=model_.max_tokens();
    const auto caps=capabilities(0);
    const char *lease=std::getenv("STRATA_DENSE_MTP_BORROW");
    if(lease&&std::string(lease)!="0"&&std::string(lease)!="1")
        throw std::invalid_argument("MTP borrow option must be0or1");
    if(caps.major==12&&caps.minor==0&&(!lease||std::string(lease)!="0")){
        // Request-scoped leases on the model's single ordered stream. Prefix
        // lives in dead input rows; cache-only catch-up does not write them.
        // Gather and decoder output reuse the dead model output at distinct
        // times. Norm/post-norm and conditioned/residual have disjoint leases.
        const auto &b=model_.shared_->buffers();
        auto &ffn=model_.shared_->shared_ffn(2);
        if(size_t(2)*t*h*4>ffn.half_weight_capacity()||!b.output||!b.post_norm||!b.residual)
            throw std::runtime_error("MTP borrowed conditioning arena geometry");
        embedding_=b.output;prefix_=model_.input_;normed_=b.post_norm;conditioned_=b.residual;
        // Legacy concat dies after input encoding; NVFP4 consumes it directly
        // through Runtime-owned encoded storage. No weight decode aliases it.
        concat_=static_cast<float *>(ffn.buffers().half_weights);
    }else{
        embedding_=floats(size_t(t)*h);concat_=floats(size_t(t)*2*h);conditioned_=floats(size_t(t)*h);
        normed_=floats(size_t(t)*h);prefix_=floats(size_t(t)*h);
    }
    seed_=floats(h);draft_seed_=floats(h);
    verify_rows_=floats(4*h);catch_rows_=floats(4*h);logits_=floats(s.n_vocab);
    device_ids_=static_cast<int *>(storage_.allocate(size_t(t)*4));picked_=static_cast<int *>(storage_.allocate(4));
    invalid_=static_cast<unsigned *>(storage_.allocate(4));greedy_bytes_=greedy_workspace(int(s.n_vocab));greedy_=storage_.allocate(greedy_bytes_);
    host_=std::make_unique<MtpPinned>();target_ids_=static_cast<int *>(storage_.allocate(56));target_flags_=reinterpret_cast<unsigned *>(target_ids_+4);
    draft_ids_=target_ids_+8;draft_flags_=reinterpret_cast<unsigned *>(target_ids_+11);
    saved_.resize(s.n_layers);
    for(size_t i=0;i<saved_.size();++i)if(s.layers[i].recurrent){
        saved_[i].state=floats(s.ssm_state*s.ssm_state*s.ssm_rank);
        saved_[i].history=floats(3*s.ssm_state*(2*s.ssm_group+s.ssm_rank));
    }
    check_cuda(cudaStreamSynchronize(stream),"own draft setup complete");
}
void NativeMtp::prepare_paged_decode(int prompt,int maximum,cudaStream_t stream){
    auto &stats=phase_kv_stats();
    if(!phase_kv_will_demote(prompt,maximum,!model_.visual_positions(),proposal_capacity_,true))return;
    const auto started=std::chrono::steady_clock::now();stats.owner=reinterpret_cast<std::uintptr_t>(this);
    stats.resident_rows=model_.kv_resident();stats.selected_tokens=model_.kv_selected();
    stats.pp_bound=model_.attention_work().ram_request_bound();
    release_graphs();stats.graphs_cleared=true;
    try{
        check_cuda(cudaStreamSynchronize(stream),"phase KV complete prefix drain");
        if(model_.kv_resident()!=40960||!ram_||!ram_->device_rows())throw std::logic_error("phase KV draft authority");
        const auto caps=capabilities(0);
        DeviceSelectedRequestScope selected(true,maximum,caps.major==12&&caps.minor==0);
        for(auto &c:model_.caches_)if(c.ram)selected.add(*c.ram);
        selected.add(*ram_);const bool keep_device=selected.ready();
        if(keep_device)selected.keep(stream);
        auto demote=[&](RamKvCache &cache){
            const auto old=cache.view();const size_t rows=size_t(old.host_capacity)*old.heads*272;
            if(old.resident!=40960||old.slots!=2560||old.selected_tokens!=8192||old.keep_queries!=3||
               !cache.device_rows()||old.host_key!=old.key||old.host_value!=old.value)
                throw std::logic_error("phase KV exact identity authority");
            if(keep_device){++stats.caches;stats.logical_capacity=old.host_capacity;return;}
            cache.preserve_to_host(maximum,stream);const auto now=cache.view();
            if(cache.device_rows()||now.key!=old.key||now.value!=old.value||now.host_key==now.key||
               now.host_value==now.value||now.host_capacity<old.host_capacity||now.resident!=old.resident)
                throw std::logic_error("phase KV demotion publication");
            ++stats.caches;++stats.mapped_caches;stats.d2h_bytes+=2*rows;
            stats.summary_bytes+=size_t(old.heads)*old.pages*512*2;stats.d2d_bytes+=size_t(old.heads)*old.pages*512*2;
            stats.logical_capacity=now.host_capacity;
        };
        for(auto &c:model_.caches_)if(c.ram){demote(*c.ram);c.key=c.ram->view().key;c.value=c.ram->view().value;++stats.main_caches;}
        demote(*ram_);++stats.draft_caches;key_=ram_->view().key;value_=ram_->view().value;
        model_.attention_work().use_ram_request_bound(0);attention_->use_ram_request_bound(0);
        model_.attention_work().use_ram_prefix_bound(0);attention_->use_ram_prefix_bound(0);
        stats.main_bound=model_.attention_work().ram_request_bound();stats.draft_bound=attention_->ram_request_bound();
        stats.main_prefix_bound=model_.attention_work().ram_prefix_bound();
        check_cuda(cudaStreamSynchronize(stream),"phase KV full authority publication");
        stats.migrated=true;++stats.epoch;
        if(keep_device)selected.publish();
    }catch(...){stats.failed=true;release_graphs();cudaStreamSynchronize(stream);throw;}
    stats.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}
void NativeMtp::reserve_fresh_cache(int required,cudaStream_t stream){
    if(required<1||required>model_.logical_context())throw std::out_of_range("fresh MTP logical KV extent");
    if(graphs_)throw std::logic_error("release MTP Graph leases before fresh cache growth");
    if(!ram_){reserve_cache(required,stream);return;}
    // Main fresh reservation has already completed. Never discard it again
    // here: other cache modes retain their existing preserving contract.
    model_.reserve_cache(required,stream);
    if(resident_kv_requested(required,model_.kv_resident()))ram_->reserve_discarding_device(required,stream);
    else ram_->reserve_discarding(required,stream);
    key_=ram_->view().key;value_=ram_->view().value;
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
