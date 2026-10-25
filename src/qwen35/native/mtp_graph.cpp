#include "mtp_state_owner.hpp"
#include "history_snapshot_policy.hpp"
#include "mtp_graph.hpp"
#include <chrono>
#include "resident_kv_policy.hpp"
#include "strata/qwen35/native/mtp_draft_link.hpp"
namespace strata::qwen35::native {
NativeMtp::~NativeMtp(){release_state_owner(&model_);graphs_.reset();parked_graphs_.reset();}
void NativeMtp::normalize_state_owner(){strata::qwen35::native::normalize_state_owner(&model_);}
void NativeMtp::release_graphs(){release_state_owner(&model_);graphs_.reset();parked_graphs_.reset();tg_request_eligible_=false;tg_request_key_.clear();mtp_prefix_graph_stats()={};graph_suspended_=false;graph_batch_={};model_.mtp_capture(nullptr);}
void NativeMtp::park_graphs(){
    normalize_state_owner();
    if(graphs_){check_cuda(cudaStreamSynchronize(graphs_->stream()),"MTP TG park drain");
        if(tg_request_eligible_&&graphs_->cacheable())parked_graphs_=std::move(graphs_);else graphs_.reset();}
    graph_suspended_=false;graph_batch_={};model_.mtp_capture(nullptr);mtp_tg_graph_stats().parked=bool(parked_graphs_);
}
void NativeMtp::prepare_tg_graph_reuse(const std::string &config,int tokens,int maximum,bool eligible,cudaStream_t stream){
    if(graphs_)throw std::logic_error("MTP TG admission requires inactive graphs");
    const auto begin=std::chrono::steady_clock::now();auto &stats=mtp_tg_graph_stats();
    stats.hit=false;stats.validation_ms=stats.rearm_ms=0;
    tg_request_eligible_=mtp_tg_graph_enabled()&&eligible&&tokens==8192&&proposal_capacity_==2&&gpu_draft_chain_&&retain_prefix_;
    // A hit never crosses grow/free/authority transfer: prove the existing early return BEFORE reserve.
    if(!ram_||!ram_->device_rows()||maximum>ram_->view().host_capacity||
       !resident_kv_requested(maximum,model_.kv_resident()))tg_request_eligible_=false;
    for(const auto &cache:model_.caches_)if(cache.key&&(!cache.ram||!cache.ram->device_rows()||
        maximum>cache.ram->view().host_capacity||maximum>cache.ram->view().resident))tg_request_eligible_=false;
    tg_request_key_=mtp_tg_graph_key(config);stats.eligible=tg_request_eligible_;
    if(parked_graphs_&&(!tg_request_eligible_||!parked_graphs_->matches(tg_request_key_,tokens,maximum,stream)))parked_graphs_.reset();
    stats.parked=bool(parked_graphs_);stats.validation_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
void NativeMtp::enable_graphs(const int *prefix,int tokens,int maximum,cudaStream_t stream){enable_graphs(prefix,tokens,maximum,stream,true);}
void NativeMtp::enable_graphs(const int *prefix,int tokens,int maximum,cudaStream_t stream,bool capture_prefix){
    if(graphs_||!model_.q8_cache()||maximum>model_.logical_context()||maximum<tokens+proposal_capacity_+1)
        throw std::invalid_argument("invalid native MTP graph reservation");
    reserve_cache(maximum,stream);
    if(!capture_prefix&&tg_request_eligible_&&parked_graphs_&&parked_graphs_->matches(tg_request_key_,tokens,maximum,stream)&&parked_graphs_->sampling_owner_==sampling_){
        const auto begin=std::chrono::steady_clock::now();graphs_=std::move(parked_graphs_);graphs_->rearm(prefix,stream);
        auto &stats=mtp_tg_graph_stats();stats.hit=true;stats.parked=false;++stats.hits;
        stats.rearm_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();return;
    }
    parked_graphs_.reset();auto &stats=mtp_tg_graph_stats();stats.parked=false;++stats.builds;++stats.misses;
    graphs_=std::make_unique<MtpGraphs>(*this,prefix,tokens,maximum,stream,capture_prefix);
}
size_t NativeMtp::graph_nodes()const{return graphs_?graphs_->nodes():0;}
size_t NativeMtp::graph_replays()const{return graphs_?graphs_->replays():0;}
double NativeMtp::graph_cold_ms()const{return graphs_?graphs_->cold_ms():0;}
unsigned MtpGraphs::parity()const{
    unsigned value=2;
    for(const auto &r:m_.model_.states_)if(r.state){
        if(value==2)value=r.current;else if(value!=r.current)throw std::runtime_error("MTP history ownership inconsistent");
    }
    if(value>1)throw std::runtime_error("missing MTP recurrent history");return value;
}
void MtpGraphs::parity(unsigned p){for(auto &r:m_.model_.states_)if(r.state)r.current=p;}
MtpGraphs::MtpGraphs(NativeMtp &m,const int *prefix,int tokens,int maximum,cudaStream_t stream,bool record_prefix)
    :m_(m),maximum_(maximum),request_key_(m.tg_request_key_),sampling_owner_(m.sampling_),stream_(stream){
    if(!prefix||tokens<2||maximum>m.model_.logical_context())throw std::invalid_argument("invalid fixed MTP graph request");
    check_cuda(cudaGetDevice(&device_),"MTP TG graph device owner");
    const auto begin=std::chrono::steady_clock::now();const unsigned original=parity();
    const auto &spec=m_.model_.spec_;const auto caps=capabilities(0);
    const bool owner=state_owner_requested()&&caps.major==12&&caps.minor==0&&m_.gpu_draft_chain_&&m_.retain_prefix_&&m_.proposal_capacity_==2&&
        m_.model_.q8_cache()&&history_snapshot_elision_requested()&&!m_.model_.visual_positions()&&spec.ssm_state==128&&spec.ssm_group==16&&spec.ssm_rank==48&&m_.model_.states_.size()<=64&&
        std::count_if(m_.model_.states_.begin(),m_.model_.states_.end(),[](const auto &v){return v.state!=nullptr;})==48;
    if(owner){std::array<StateOwnerRow,48> bindings{};size_t at=0;
        for(size_t i=0;i<m_.model_.states_.size();++i)if(m_.model_.states_[i].state){auto &r=m_.model_.states_[i];
            bindings[at++]={int(i),&r.state,&r.current,r.state,m_.saved_[i].state};}
        prepare_state_owner(&m_.model_,&m_,bindings,original,stream,&m_.counters_.D2D_bytes);
    }else release_state_owner(&m_.model_);
    const auto counts=m_.counters_; // Preserve real cold normalization bytes; discard only capture bookkeeping.

    batch_.start=static_cast<int *>(cursor_storage_.allocate(4));
    batch_.cursor=static_cast<DecodeCursor *>(cursor_storage_.allocate(sizeof(DecodeCursor)));
    // The policy scans the same prefix as valid grows, so its maximum selected
    // count is attained at maximum. Avoid a grid of empty dynamic CTAs.
    batch_.maximum_partitions=m_.gpu_draft_chain_?
        decode_partition_count(m_.model_.kv_window()?std::min(maximum,m_.model_.kv_window()):maximum,int(m_.model_.spec_.n_head),m_.attention_->decode_rounding()):(maximum+255)/256;
    m_.host_->p->position=tokens;
    check_cuda(cudaMemcpyAsync(const_cast<int *>(batch_.start),&m_.host_->p->position,4,cudaMemcpyHostToDevice,stream),"MTP graph initial cursor");
    check_cuda(cudaMemsetAsync(batch_.cursor,0,sizeof(DecodeCursor),stream),"MTP graph cursor setup");
    check_cuda(cudaStreamSynchronize(stream),"MTP graph stable cursor lease");
    mtp_exec_stats()={};mtp_prefix_graph_stats()={};mtp_prefix_graph_stats().initial_omitted=!record_prefix;
    const bool deferred=defer_mtp_exec(m_.gpu_draft_chain_,m_.retain_prefix_,m_.proposal_capacity_);
    prefix_input_=prefix;prefix_tokens_=tokens;prefix_eager_=!deferred;
    m_.graph_building_=true;
    try{
        m_.graph_batch_={};
        if(record_prefix)capture_prefix(false);
        for(int p=0;p<2;++p)for(int which=0;which<5;++which){
            const int count=which==4?2:which+1;
            if(count>m_.proposal_capacity_+1)continue;
            parity(unsigned(p));m_.graph_batch_=batch_;
            target_[p][which].capture(stream,[&]{
                StateOwnerCapture route(&m_.model_,unsigned(p),(which>=1&&which<=2)?1:0);
                if(which>=1&&which<=3)m_.snapshot(stream);
                m_.model_.mtp_capture(&batch_);
                m_.enqueue_target(m_.device_ids_,count,tokens,stream,m_.retain_prefix_&&(which>=1&&which<=3));m_.model_.mtp_capture(nullptr);
            },8192,!deferred||which==2);
        }
        for(int p=0;p<2;++p){
            for(auto &save:m_.saved_)if(save.state)save.parity=unsigned(p);
            rewind_[p].capture(stream,[&]{StateOwnerCapture route(&m_.model_,unsigned(p),0);m_.rewind(stream);},8192,!deferred);
        }
        if(m_.retain_prefix_)for(int p=0;p<2;++p)for(int count=1;count<=m_.proposal_capacity_;++count){
            parity(unsigned(p));prefix_restore_[p][count-1].capture(stream,[&]{StateOwnerCapture route(&m_.model_,unsigned(p),2);m_.restore_prefix(count,stream);});
        }
        for(int step=0;step<m_.proposal_capacity_;++step){
            m_.graph_batch_=batch_;m_.graph_offset_=step;
            draft_[step].capture(stream,[&]{
                m_.draft(m_.device_ids_+step,step?m_.draft_seed_:m_.seed_,tokens+step,stream,unsigned(step));
                if(m_.gpu_draft_chain_)mtp_draft_link(m_.picked_,m_.invalid_,int(m_.model_.spec_.n_vocab),
                    m_.device_ids_+step+1,m_.draft_ids_+step,m_.draft_flags_+step,stream);
            });
        }
        for(int count=1;count<=m_.proposal_capacity_+1;++count){
            m_.graph_batch_=batch_;m_.graph_offset_=0;
            commit_[count-1].capture(stream,[&]{m_.enqueue_commit(count,tokens,stream);});
        }
    }catch(...){m_.model_.mtp_capture(nullptr);m_.graph_building_=false;parity(original);m_.counters_=counts;throw;}
    m_.graph_batch_={};m_.graph_building_=false;parity(original);m_.counters_=counts;
    cold_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
MtpGraphs::~MtpGraphs(){cudaStreamSynchronize(stream_);}
bool MtpGraphs::matches(const std::string &key,int tokens,int maximum,cudaStream_t stream)const{
    int device=-1;check_cuda(cudaGetDevice(&device),"MTP TG current device owner");
    return cacheable()&&device==device_&&key==request_key_&&tokens==prefix_tokens_&&maximum==maximum_&&stream==stream_;
}
void MtpGraphs::rearm(const int *prefix,cudaStream_t stream){
    m_.normalize_state_owner();
    if(!prefix||!cacheable()||stream!=stream_)throw std::logic_error("MTP TG rearm lease");
    // PP has already rebuilt all current-request state. Do not reset its actual parity.
    (void)parity();prefix_input_=prefix;m_.graph_batch_={};m_.graph_offset_=0;m_.graph_building_=false;
    m_.host_->p->position=prefix_tokens_;
    check_cuda(cudaMemcpyAsync(const_cast<int*>(batch_.start),&m_.host_->p->position,4,cudaMemcpyHostToDevice,stream),"MTP TG fresh cursor");
    check_cuda(cudaMemsetAsync(batch_.cursor,0,sizeof(DecodeCursor),stream),"MTP TG fresh decode cursor");
    for(auto &g:draft_)g.reset_replays();for(auto &row:target_)for(auto &g:row)g.reset_replays();
    for(auto &g:rewind_)g.reset_replays();for(auto &g:commit_)g.reset_replays();
    for(auto &row:prefix_restore_)for(auto &g:row)g.reset_replays();cold_ms_=0;
    mtp_exec_stats()={};mtp_prefix_graph_stats()={};mtp_prefix_graph_stats().initial_omitted=true;
}
void MtpGraphs::begin(int position,cudaStream_t stream){
    if(stream!=stream_||position<1||position>=maximum_)throw std::invalid_argument("MTP graph request lease changed");
    m_.host_->p->position=position;
    check_cuda(cudaMemcpyAsync(const_cast<int *>(batch_.start),&m_.host_->p->position,4,cudaMemcpyHostToDevice,stream),"MTP graph actual cursor");
    m_.counters_.H2D_bytes+=4;
}
void MtpGraphs::capture_prefix(bool lazy){
    if(prefix_.nodes())throw std::logic_error("MTP prefix graph already recorded");
    const auto started=std::chrono::steady_clock::now();const auto counts=m_.counters_;const auto batch=m_.graph_batch_;
    const auto before=mtp_exec_stats();const unsigned original=parity();const bool building=m_.graph_building_;
    m_.graph_building_=true;m_.graph_batch_={};
    m_.model_.suppress_prefill_progress(true);
    try{
        const int chunks=(prefix_tokens_+m_.model_.max_tokens()-1)/m_.model_.max_tokens();
        const size_t budget=size_t(chunks)*(m_.model_.attention_work().pp_blocked()?16384:8192);
        prefix_.capture(stream_,[&]{m_.model_.reset_prefill(stream_);m_.prefill(prefix_input_,prefix_tokens_,stream_);},budget,prefix_eager_);
        prefix_parity_=parity();
    }catch(...){m_.model_.suppress_prefill_progress(false);m_.model_.mtp_capture(nullptr);m_.graph_batch_=batch;m_.graph_building_=building;parity(original);m_.counters_=counts;throw;}
    m_.model_.suppress_prefill_progress(false);
    m_.graph_batch_=batch;m_.graph_building_=building;parity(original);m_.counters_=counts;
    auto &stats=mtp_prefix_graph_stats();const auto after=mtp_exec_stats();stats.ready=true;stats.lazy_builds+=lazy;
    stats.record_ms+=after.record_ms-before.record_ms;stats.instantiate_ms+=after.instantiate_ms-before.instantiate_ms;
    stats.cold_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}
void MtpGraphs::prefix(cudaStream_t stream){
    if(stream!=stream_)throw std::invalid_argument("MTP prefix graph stream lease changed");
    if(!prefix_.nodes())capture_prefix(true);
    const auto before=mtp_exec_stats();prefix_.launch(stream);const double inst=mtp_exec_stats().instantiate_ms-before.instantiate_ms;
    mtp_prefix_graph_stats().instantiate_ms+=inst;mtp_prefix_graph_stats().cold_ms+=inst;
    parity(prefix_parity_);++m_.counters_.graph_PP;
}
void MtpGraphs::draft(int step,cudaStream_t stream){
    if(step<0||step>=m_.proposal_capacity_)throw std::invalid_argument("MTP graph draft step");draft_[step].launch(stream);++m_.counters_.graph_draft;
}
void MtpGraphs::target(int count,bool preserve,cudaStream_t stream){
    if(count<1||count>m_.proposal_capacity_+1)throw std::invalid_argument("MTP graph verification span");
    const auto p=parity();const int index=count==2&&!preserve?4:count-1;
    auto *owner=state_owner(&m_.model_);const bool outstate=owner&&preserve&&count>=2&&count<=3;
    if(outstate)owner->begin(p);else m_.normalize_state_owner();
    if(preserve){
        for(auto &save:m_.saved_)if(save.state)save.parity=p;
        const auto &s=m_.model_.spec_;
        if(!outstate)m_.counters_.D2D_bytes+=size_t(48)*(s.ssm_state*s.ssm_state*s.ssm_rank+3*s.ssm_state*(2*s.ssm_group+s.ssm_rank))*4;
    }
    target_[p][index].launch(stream);parity(p^1);if(outstate)owner->bind(p^1);
    if(preserve)++m_.counters_.graph_verify;else ++m_.counters_.graph_replay;
}
void MtpGraphs::rewind(cudaStream_t stream){
    unsigned p=2;for(const auto &s:m_.saved_)if(s.state){p=s.parity;break;}
    if(p>1)throw std::runtime_error("MTP saved history owner missing");
    if(auto *owner=state_owner(&m_.model_);owner&&owner->pending){owner->rollback();parity(p);++m_.counters_.graph_rewind;return;}
    m_.normalize_state_owner();rewind_[p].launch(stream);parity(p);++m_.counters_.graph_rewind;
    const auto &s=m_.model_.spec_;
    m_.counters_.D2D_bytes+=size_t(48)*(s.ssm_state*s.ssm_state*s.ssm_rank+3*s.ssm_state*(2*s.ssm_group+s.ssm_rank))*4;
}
void MtpGraphs::commit(int count,cudaStream_t stream){
    if(count<1||count>m_.proposal_capacity_+1)throw std::invalid_argument("MTP graph commit span");commit_[count-1].launch(stream);if(auto *owner=state_owner(&m_.model_))owner->commit();++m_.counters_.graph_commit;
}
void MtpGraphs::restore_prefix(int count,cudaStream_t stream){
    if(!m_.retain_prefix_||count<1||count>m_.proposal_capacity_)throw std::invalid_argument("MTP retained prefix span");
    prefix_restore_[parity()][count-1].launch(stream);
    m_.counters_.D2D_bytes+=m_.recurrent_bytes();m_.counters_.prefix_restored_rows+=count;
    ++m_.counters_.graph_prefix_restore;
}
size_t MtpGraphs::nodes()const{
    size_t result=prefix_.nodes();for(const auto &g:draft_)result+=g.nodes();
    for(const auto &row:target_)for(const auto &g:row)result+=g.nodes();
    if(m_.retain_prefix_)for(const auto &row:prefix_restore_)for(const auto &g:row)result+=g.nodes();
    for(const auto &g:rewind_)result+=g.nodes();for(const auto &g:commit_)result+=g.nodes();return result;
}
size_t MtpGraphs::replays()const{
    size_t result=prefix_.replays();for(const auto &g:draft_)result+=g.replays();
    for(const auto &row:target_)for(const auto &g:row)result+=g.replays();
    if(m_.retain_prefix_)for(const auto &row:prefix_restore_)for(const auto &g:row)result+=g.replays();
    for(const auto &g:rewind_)result+=g.replays();for(const auto &g:commit_)result+=g.replays();return result;
}
}
