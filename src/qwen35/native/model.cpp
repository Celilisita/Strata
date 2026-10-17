#include "strata/qwen35/native/model.hpp"
#include "strata/qwen35/native/ops.hpp"
#include "small_prefill_q6_planes.hpp"
#include <chrono>
#include <algorithm>

namespace strata::qwen35::native {
DenseModel::DenseModel(const strata::qwen35::Spec &s,const DenseWeights &w,DenseLimits l,cudaStream_t stream)
    :spec_(s),weights_(w),limits_(l){
    if(w.tiled_ffn()&&(!l.native32||!l.qkv_byte_planes))throw std::invalid_argument("tiled FFN inventory requires admitted PP32 byte schedule");
    if(l.fused_pack32&&(!l.native32||!l.qkv_byte_planes))throw std::invalid_argument("fused PP32 packing requires byte QKV schedule");
    if(l.folded_ring_ffn&&(!l.folded_ffn||l.folded_compact_ffn||l.folded_widewarp_ffn))throw std::invalid_argument("ring model requires exclusive folded FFN");
    if(l.folded_widewarp_ffn&&(!l.folded_ffn||l.folded_compact_ffn))throw std::invalid_argument("widewarp model requires exclusive folded FFN");
    if(l.folded_gray_ffn&&!l.folded_compact_ffn)throw std::invalid_argument("gray model requires compact folded FFN");
    if(l.folded_compact_ffn&&!l.folded_ffn)throw std::invalid_argument("compact model requires folded FFN");
    if(l.folded_ffn&&(!l.joint_ffn||l.bank_ffn||l.operand_lookahead||l.warp64_pair||l.token256_pair))throw std::invalid_argument("folded model requires exclusive joint FFN");
    if(l.bank_ffn&&(!l.joint_ffn||l.operand_lookahead||l.warp64_pair||l.token256_pair))throw std::invalid_argument("bank-layout model requires exclusive joint FFN");
    if(l.operand_lookahead&&(!l.joint_projections||l.warp64_pair||l.token256_pair))throw std::invalid_argument("operand lookahead requires exclusive joint projections");
    if(l.warp64_pair&&(!l.joint_projections||l.token256_pair))throw std::invalid_argument("warp64 requires exclusive joint projections");
    if(l.token256_pair&&!l.joint_projections)throw std::invalid_argument("token256 model requires admitted joint projections");
    if(l.joint_projections&&!l.joint_ffn)throw std::invalid_argument("joint projections require long joint FFN admission");
    if(l.joint_ffn&&(l.qkv_planes||l.max_tokens<128))throw std::invalid_argument("joint long PP requires unleased raw inventory and wide capacity");
    if(l.wide_grid&&l.native32)throw std::invalid_argument("wide-grid candidate and native32 admission are separate");
    if(l.qkv_byte_planes&&!l.qkv_planes)throw std::invalid_argument("byte QKV view requires plane admission");
    if(l.qkv_planes&&(!l.native32||l.max_tokens<32))throw std::invalid_argument("QKV planes require native32 capacity");
    if(l.context!=262144 || l.context!=int(s.context_length) || l.batch!=8192 || l.ubatch!=8192 ||
       l.max_tokens<2 || l.max_tokens>l.ubatch || s.n_layers!=64 || !s.layers[0].recurrent || s.layers[3].recurrent ||
       s.head_dim!=256 || s.ssm_conv!=4 || w.recurrent.size()!=s.layers.size() || w.attention.size()!=s.layers.size())
        throw std::invalid_argument("unsupported fixed Dense prefill configuration");
    RecurrentSchedule rp;const int parts[]={1,3,20,20,2};
    for(int i=0;i<5;++i)rp.projections[i].partitions=parts[i];
    rp.wide_grid=l.wide_grid||l.joint_ffn;
    shared_=std::make_unique<RecurrentLayer>(w.recurrent[0],RecurrentShape{int(s.n_embd),int(s.ssm_state),
        int(s.ssm_group),int(s.ssm_rank),l.max_tokens,float(s.rms_epsilon),int(2*s.n_head*s.head_dim),
        20*size_t(s.n_head_kv*s.head_dim),true},l.joint_ffn?4:8,rp);
    AttentionSchedule ap;const int attn_parts[]={3,20,20,2};
    for(int i=0;i<4;++i)ap.projections[i].partitions=attn_parts[i];
    const auto caps=capabilities(0);ap.rounding={32,2,caps.multiprocessors,256};
    attention_=std::make_unique<AttentionLayer>(*shared_,w.attention[3],AttentionShape{int(s.n_embd),int(s.head_dim),
        int(s.n_head),int(s.n_head_kv),int(s.rope_dim),float(s.rms_epsilon),float(s.rope_base)},ap);
    if(l.native32)shared_->enable_native32(stream);
    if(l.joint_ffn)shared_->shared_ffn(l.max_tokens).enable_joint_tiles();
    if(l.joint_projections)shared_->enable_joint_projections();
    if(l.folded_ffn)shared_->shared_ffn(l.max_tokens).enable_folded_pair();
    if(l.folded_ring_ffn)shared_->shared_ffn(l.max_tokens).enable_folded_ring_pair();
    if(l.folded_widewarp_ffn)shared_->shared_ffn(l.max_tokens).enable_folded_wide_warp();
    if(l.folded_compact_ffn)shared_->shared_ffn(l.max_tokens).enable_folded_compact_pair();
    if(l.folded_gray_ffn)shared_->shared_ffn(l.max_tokens).enable_folded_gray_pair();
    if(l.bank_ffn)shared_->shared_ffn(l.max_tokens).enable_bank_tiles();
    if(l.operand_lookahead)shared_->shared_ffn(l.max_tokens).enable_operand_lookahead();
    if(l.warp64_pair)shared_->shared_ffn(l.max_tokens).enable_warp64_pair();
    if(l.token256_pair)shared_->shared_ffn(l.max_tokens).enable_token256_pair();
    if(l.fused_pack32)shared_->enable_fused_pack32();
    states_.resize(s.n_layers);caches_.resize(s.n_layers);
    const size_t history=3*s.ssm_state*(2*s.ssm_group+s.ssm_rank)*sizeof(float);
    const size_t state=s.ssm_state*s.ssm_state*s.ssm_rank*sizeof(float);
    if(l.kv_keep_queries&&!l.kv_ram_tokens)throw std::invalid_argument("reuse horizon requires RAM KV");
    if(l.kv_select_tokens&&!l.kv_ram_tokens)throw std::invalid_argument("history budget requires RAM KV");
    if(l.kv_ram_tokens&&(l.kv_ram_tokens!=32768||l.kv_window_tokens||!l.kv_q8))
        throw std::invalid_argument("RAM history requires Q8 and exclusive32768resident cache");
    if(l.kv_window_tokens&&(l.kv_window_tokens!=32768||!l.kv_q8))
        throw std::invalid_argument("sliding selection requires Q8 and explicit32768window");
    cache_capacity_=l.kv_initial_tokens?l.kv_initial_tokens:(l.kv_window_tokens?8192:l.context);
    if(l.kv_window_tokens)cache_capacity_=std::min(cache_capacity_,l.kv_window_tokens+l.max_tokens);
    if(l.kv_ram_tokens)cache_capacity_=l.kv_ram_tokens;
    if(cache_capacity_<2||cache_capacity_>l.context)throw std::invalid_argument("invalid initial KV capacity");
    const size_t cache=size_t(cache_capacity_)*s.n_head_kv*(s.head_dim/32)*(l.kv_q8?34:18);
    auto zero=[&](size_t n){void *p=storage_.allocate(n);check_cuda(cudaMemsetAsync(p,0,n,stream),"own state/cache setup reset");return p;};
    for(size_t i=0;i<s.n_layers;++i){
        if(s.layers[i].recurrent){states_[i].history[0]=static_cast<float *>(zero(history));
            states_[i].history[1]=static_cast<float *>(zero(history));states_[i].state=static_cast<float *>(zero(state));}
        else if(l.kv_ram_tokens){auto &c=caches_[i];
            c.ram=std::make_unique<RamKvCache>(l.kv_initial_tokens?l.kv_initial_tokens:8192,l.kv_ram_tokens,int(s.n_head_kv),stream,kv_selected(),kv_reuse_horizon());
            c.key=c.ram->view().key;c.value=c.ram->view().value;
        }else{caches_[i].key=zero(cache);caches_[i].value=zero(cache);}
    }
    input_=static_cast<float *>(storage_.allocate(size_t(l.max_tokens)*s.n_embd*4));
    positions_=static_cast<int *>(storage_.allocate(size_t(l.max_tokens)*4));
    final_norm_=static_cast<float *>(storage_.allocate(s.n_embd*4));
    logits_=static_cast<float *>(storage_.allocate(s.n_vocab*4));
    picked_=static_cast<int *>(storage_.allocate(4));invalid_=static_cast<unsigned *>(storage_.allocate(4));
    greedy_bytes_=greedy_workspace(int(s.n_vocab));greedy_=storage_.allocate(greedy_bytes_);
    batch_logits_=static_cast<float *>(storage_.allocate(size_t(3)*s.n_vocab*4));
    batch_greedy_=storage_.allocate(3*greedy_bytes_);
    if(l.qkv_planes){
        const auto cold=std::chrono::steady_clock::now();qkv_views_.resize(s.n_layers);
        const auto &ffn=shared_->shared_ffn(32);
        const size_t row_bytes=size_t(ffn.intermediate_features())*sizeof(float);
        const size_t live=32*row_bytes,capacity=size_t(l.max_tokens)*row_bytes;
        unsigned char *arenas[]={reinterpret_cast<unsigned char *>(ffn.buffers().gate),
                                 reinterpret_cast<unsigned char *>(ffn.buffers().up),
                                 reinterpret_cast<unsigned char *>(ffn.buffers().activation)};
        size_t cursor=live;int arena=0;
        for(size_t i=0;i<s.n_layers;++i)if(s.layers[i].recurrent&&w.recurrent[i].qkv.quant==Quant::Q6K){
            const Matrix raw=w.recurrent[i].qkv;const size_t bytes=l.qkv_byte_planes?q6_byte_planes_bytes(raw):matrix_bytes(raw);
            if(bytes>capacity-cursor){++arena;cursor=live;}
            if(arena>=3||bytes>capacity-cursor)throw std::runtime_error("PP32 dead workspace cannot hold QKV views");
            auto view=l.qkv_byte_planes?prepare_q6_byte_planes(raw,arenas[arena]+cursor,stream):prepare_q6_planes(raw,arenas[arena]+cursor,stream);cursor+=bytes;
            qkv_views_[i]={view.data,view.bytes};qkv_plane_bytes_+=view.bytes;++qkv_plane_count_;
        }
        check_cuda(cudaStreamSynchronize(stream),"immutable QKV plane setup lifetime");
        qkv_plane_cold_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold).count();
    }
    check_cuda(cudaStreamSynchronize(stream),"own prefill setup completion");
    size_t free=0,total=0;check_cuda(cudaMemGetInfo(&free,&total),"own prefill setup memory reserve");
    if(free<DeviceStorage::reserve_bytes())throw std::runtime_error("own Dense setup lacks configured headroom");
}
void DenseModel::reset_prefill(cudaStream_t stream){
    for(auto &c:caches_)if(c.ram)c.ram->reset(stream);
    const size_t history=3*spec_.ssm_state*(2*spec_.ssm_group+spec_.ssm_rank)*sizeof(float);
    const size_t state=spec_.ssm_state*spec_.ssm_state*spec_.ssm_rank*sizeof(float);
    for(size_t i=0;i<states_.size();++i)if(spec_.layers[i].recurrent){
        auto &r=states_[i];r.current=0;
        check_cuda(cudaMemsetAsync(r.state,0,state,stream),"own fresh request state reset");
        for(auto *h:r.history)check_cuda(cudaMemsetAsync(h,0,history,stream),"own fresh request history reset");
    }
}
float *DenseModel::embed(const int *ids,int start,int tokens,cudaStream_t stream){
    if(tokens<1 || tokens>limits_.max_tokens || start<0 || start>limits_.context-tokens)
        throw std::invalid_argument("invalid Dense input token span");
    reserve_cache(start+tokens,stream);
    gather_embedding(weights_.embedding,ids,input_,tokens,stream);
    visual_embedding(input_,start,tokens,stream);
    if(decode_capture_){
        if(tokens!=1)throw std::invalid_argument("device cursor is single-token only");
        set_decode_position(positions_,decode_capture_->cursor,int(spec_.n_head),attention_->decode_rounding(),stream);
    }else if(mtp_capture_)mtp_positions(positions_,mtp_capture_->start,0,tokens,0,mtp_capture_->cursor,int(spec_.n_head),attention_->decode_rounding(),stream);
    else set_positions(positions_,start,tokens,stream);
    return input_;
}
float *DenseModel::layer(int index,const float *hidden,int start,int tokens,cudaStream_t stream){
    if(limits_.qkv_planes&&tokens!=32&&tokens!=1)
        throw std::invalid_argument("QKV workspace view lease permits PP32 and single-token continuation only");
    if(index<0 || index>=int(spec_.n_layers) || !hidden || tokens<1 || tokens>limits_.max_tokens ||
       start<0 || start>limits_.context-tokens)throw std::invalid_argument("invalid own prefill layer invocation");
    if(spec_.layers[index].recurrent){
        auto &r=states_[index];shared_->rebind(weights_.recurrent[index]);
        if(limits_.qkv_planes&&tokens==32&&qkv_views_[index].data)
            shared_->borrow_native32_qkv_planes(qkv_views_[index].data,qkv_views_[index].bytes,stream);
        shared_->execute({hidden,r.history[r.current],r.state},{r.history[r.current^1],r.state,mtp_prefix_?mtp_prefix_[index]:GdnPrefix{}},tokens,stream);
        r.current^=1;
    }else{
        auto &c=caches_[index];attention_->rebind(weights_.attention[index]);
        AttentionCache cache{c.key,c.value,start,start+tokens,cache_capacity_,
            decode_capture_?decode_capture_->cursor:nullptr,decode_capture_?decode_capture_->partitions:0,limits_.kv_q8};
        if(mtp_capture_){cache.cursor=tokens==1?mtp_capture_->cursor:nullptr;
            cache.graph_partitions=mtp_capture_->maximum_partitions;cache.graph_start=mtp_capture_->start;}
        cache.window=cache_capacity_>limits_.kv_window_tokens?limits_.kv_window_tokens:0;
        cache.ram=c.ram?&c.ram->view():nullptr;cache.full_history=prefilling_;
        attention_->execute(hidden,positions_,cache,tokens,stream);
    }
    return shared_->buffers().output;
}
CacheSnapshot DenseModel::cache_snapshot(int index) const{
    if(index<0 || index>=int(spec_.n_layers) || spec_.layers[index].recurrent)
        throw std::invalid_argument("invalid cache snapshot layer");
    return {caches_[index].key,caches_[index].value,cache_capacity_};
}
void DenseModel::prefill(const int *ids,int tokens,cudaStream_t stream){
    if(tokens>limits_.max_tokens){
        for(int start=0;start<tokens;start+=limits_.max_tokens){
            const int n=std::min(limits_.max_tokens,tokens-start);
            prefill_chunk(ids+start,start,n,start+n==tokens,stream);
        }
        return;
    }
    prefill_chunk(ids,0,tokens,true,stream);
}
void DenseModel::prefill_chunk(const int *ids,int start,int tokens,bool pick_last,cudaStream_t stream){
    const bool previous=prefilling_;prefilling_=true;
    try{
        const float *hidden=embed(ids,start,tokens,stream);
        for(int i=0;i<int(spec_.n_layers);++i)hidden=layer(i,hidden,start,tokens,stream);
        if(pick_last)head(hidden+size_t(tokens-1)*spec_.n_embd,stream);
    }catch(...){prefilling_=previous;throw;}
    prefilling_=previous;
}

void DenseModel::validate_captured_prefill(bool complete){
    if(!limits_.native32)throw std::invalid_argument("captured PP requires admitted native32 schedule");
    for(const auto &r:states_)if(r.state&&r.current)throw std::invalid_argument("captured PP requires fresh reset state");
    if(complete)for(auto &r:states_)if(r.state)r.current=1;
}
void DenseModel::decode(int position,cudaStream_t stream){
    const float *hidden=embed(picked_,position,1,stream);
    for(int i=0;i<int(spec_.n_layers);++i)hidden=layer(i,hidden,position,1,stream);
    head(hidden,stream);
}
void DenseModel::validate_captured_decode_pair() const{
    if(!limits_.native32){
        int device=0;check_cuda(cudaGetDevice(&device),"Dense graph admission device");
        const auto caps=capabilities(device);
        // Single-token replay uses raw vector operators and the same state/KV
        // leases; it does not borrow Volta PP32's prepared weight/input planes.
        if(caps.major<8||limits_.qkv_planes||limits_.joint_ffn||weights_.tiled_ffn())
            throw std::invalid_argument("decode graph requires native32 or an unleased SM80+ staged profile");
    }
    unsigned parity=2;
    for(const auto &r:states_)if(r.state){
        if(parity==2)parity=r.current;
        else if(r.current!=parity)throw std::invalid_argument("decode pair history ownership inconsistent");
    }
    if(parity>1)throw std::invalid_argument("decode pair history owner missing");
    const int count=decode_partition_count(limits_.kv_window_tokens?limits_.kv_window_tokens:(limits_.kv_ram_tokens?kv_selected():limits_.context),int(spec_.n_head),attention_->decode_rounding());
    if(size_t(spec_.n_head)*count*258*4>shared_->shared_ffn(2).half_weight_capacity())
        throw std::invalid_argument("decode graph maximum KV reduction exceeds shared arena");
}
std::vector<ContinuationView> DenseModel::continuation_views(int valid) const{
    if(valid<1||valid>limits_.context||(!limits_.kv_window_tokens&&valid>cache_logical_capacity()))throw std::invalid_argument("invalid continuation snapshot extent");
    std::vector<ContinuationView> out;
    const size_t history=3*spec_.ssm_state*(2*spec_.ssm_group+spec_.ssm_rank)*4;
    const size_t state=spec_.ssm_state*spec_.ssm_state*spec_.ssm_rank*4;
    const size_t cache=size_t(std::min(valid,cache_logical_capacity()))*spec_.n_head_kv*(spec_.head_dim/32)*(limits_.kv_q8?34:18);
    for(int i=0;i<int(spec_.n_layers);++i){
        if(spec_.layers[i].recurrent){out.push_back({states_[i].state,state,i,0,false});out.push_back({states_[i].history[states_[i].current],history,i,1,false});}
        else{
            if(caches_[i].ram){const auto &v=caches_[i].ram->view();
                out.push_back({v.host_key,cache,i,2,true});out.push_back({v.host_value,cache,i,3,true});
            }else if(limits_.kv_window_tokens){
                const int live=std::min(valid,limits_.kv_window_tokens);
                const size_t row=spec_.n_head_kv*(spec_.head_dim/32)*34;
                const int first=(valid-live)%cache_capacity_;
                out.push_back({caches_[i].key,size_t(live)*row,i,2,true,row,cache_capacity_,first});
                out.push_back({caches_[i].value,size_t(live)*row,i,3,true,row,cache_capacity_,first});
            }else{out.push_back({caches_[i].key,cache,i,2,true});out.push_back({caches_[i].value,cache,i,3,true});}
        }
    }
    out.push_back({positions_,4,-1,4,true});return out;
}
void DenseModel::head(const float *last_hidden,cudaStream_t stream){
    rms_norm(last_hidden,weights_.output_norm,final_norm_,int(spec_.n_embd),1,float(spec_.rms_epsilon),stream);
    void *quant=shared_->shared_ffn(1).buffers().quant_input;
    vector_quantize(final_norm_,quant,int(spec_.n_embd),stream);vector_linear(weights_.head,quant,logits_,stream);
    greedy_pick(logits_,int(spec_.n_vocab),greedy_,greedy_bytes_,picked_,invalid_,stream,token_mask_);
    if(decode_capture_)record_decode_token(picked_,invalid_,*decode_capture_,stream);
}
}
