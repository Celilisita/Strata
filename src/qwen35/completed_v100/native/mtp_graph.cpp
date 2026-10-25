#include "mtp_graph.hpp"
#include "mtp_abort.hpp"
#include <chrono>
#include "strata/qwen35/completed_v100/native/mtp_draft_link.hpp"
namespace strata::completed_v100::qwen35::native {
NativeMtp::~NativeMtp(){graphs_.reset();mtp_abort::clear(*this);}
void NativeMtp::release_graphs(){graphs_.reset();graph_suspended_=false;graph_batch_={};model_.mtp_capture(nullptr);}
void NativeMtp::park_graphs(){
    if(graphs_&&prefix_graph_enabled())throw std::logic_error("Cannot park a captured prefix graph");
    graph_suspended_=bool(graphs_);graph_batch_={};model_.mtp_capture(nullptr);
}
void NativeMtp::enable_graphs(const int *prefix,int tokens,int maximum,cudaStream_t stream){
    enable_graphs(prefix,tokens,maximum,stream,true);
}
void NativeMtp::enable_graphs(const int *prefix,int tokens,int maximum,cudaStream_t stream,bool capture_prefix){
    if(graphs_||!model_.q8_cache()||maximum>model_.logical_context()||maximum<tokens+3)
        throw std::invalid_argument("invalid native MTP graph reservation");
    reserve_cache(maximum,stream);
    graphs_=std::make_unique<MtpGraphs>(*this,prefix,tokens,maximum,stream,capture_prefix);
}
bool NativeMtp::prefix_graph_enabled()const{return graphs_&&graphs_->has_prefix();}
size_t NativeMtp::prefix_graph_nodes()const{return graphs_?graphs_->prefix_nodes():0;}
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
MtpGraphs::MtpGraphs(NativeMtp &m,const int *prefix,int tokens,int maximum,cudaStream_t stream,bool capture_prefix)
    :m_(m),maximum_(maximum),stream_(stream){
    if(!prefix||tokens<2||maximum>m.model_.logical_context())throw std::invalid_argument("invalid fixed MTP graph request");
    const auto begin=std::chrono::steady_clock::now();const auto counts=m_.counters_;const unsigned original=parity();
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
    m_.graph_building_=true;
    try{
        m_.graph_batch_={};
        if(capture_prefix){
            const int chunks=(tokens+m_.model_.max_tokens()-1)/m_.model_.max_tokens();
            const size_t pp_budget=size_t(chunks)*(m_.model_.attention_work().pp_blocked()?16384:8192);
            prefix_.capture(stream,[&]{m_.model_.reset_prefill(stream);m_.prefill(prefix,tokens,stream);},pp_budget);
            prefix_parity_=parity();
        }
        for(int p=0;p<2;++p)for(int which=0;which<4;++which){
            const int count=which==3?2:which+1;parity(unsigned(p));m_.graph_batch_=batch_;
            target_[p][which].capture(stream,[&]{
                if(which==1||which==2)m_.snapshot(stream);
                m_.model_.mtp_capture(&batch_);
                m_.enqueue_target(m_.device_ids_,count,tokens,stream,m_.retain_prefix_&&(which==1||which==2));m_.model_.mtp_capture(nullptr);
            });
        }
        if(!mtp_abort::selected(m_,stream))for(int p=0;p<2;++p){
            for(auto &save:m_.saved_)if(save.state)save.parity=unsigned(p);
            rewind_[p].capture(stream,[&]{m_.rewind(stream);});
        }
        if(m_.retain_prefix_)for(int p=0;p<2;++p)for(int count=1;count<=2;++count){
            parity(unsigned(p));prefix_restore_[p][count-1].capture(stream,[&]{m_.restore_prefix(count,stream);});
        }
        for(int step=0;step<2;++step){
            m_.graph_batch_=batch_;m_.graph_offset_=step;
            draft_[step].capture(stream,[&]{
                m_.draft(m_.device_ids_+step,step?m_.draft_seed_:m_.seed_,tokens+step,stream);
                if(m_.gpu_draft_chain_)mtp_draft_link(m_.picked_,m_.invalid_,int(m_.model_.spec_.n_vocab),
                    m_.device_ids_+step+1,m_.draft_ids_+step,m_.draft_flags_+step,stream);
            });
        }
        for(int count=1;count<=3;++count){
            m_.graph_batch_=batch_;m_.graph_offset_=0;
            commit_[count-1].capture(stream,[&]{m_.enqueue_commit(count,tokens,stream);});
        }
    }catch(...){m_.model_.mtp_capture(nullptr);m_.graph_building_=false;parity(original);m_.counters_=counts;throw;}
    m_.graph_batch_={};m_.graph_building_=false;parity(original);m_.counters_=counts;
    cold_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
MtpGraphs::~MtpGraphs(){cudaStreamSynchronize(stream_);}
void MtpGraphs::begin(int position,cudaStream_t stream){
    if(stream!=stream_||position<1||position>=maximum_)throw std::invalid_argument("MTP graph request lease changed");
    m_.host_->p->position=position;
    check_cuda(cudaMemcpyAsync(const_cast<int *>(batch_.start),&m_.host_->p->position,4,cudaMemcpyHostToDevice,stream),"MTP graph actual cursor");
    m_.counters_.H2D_bytes+=4;
}
void MtpGraphs::prefix(cudaStream_t stream){prefix_.launch(stream);parity(prefix_parity_);++m_.counters_.graph_PP;}
void MtpGraphs::draft(int step,cudaStream_t stream){
    if(step<0||step>1)throw std::invalid_argument("MTP graph draft step");draft_[step].launch(stream);++m_.counters_.graph_draft;
}
void MtpGraphs::target(int count,bool preserve,cudaStream_t stream){
    if(count<1||count>3)throw std::invalid_argument("MTP graph verification span");
    const auto p=parity();const int index=count==2&&!preserve?3:count-1;
    if(preserve&&!mtp_abort::selected(m_,stream)){
        for(auto &save:m_.saved_)if(save.state)save.parity=p;
        const auto &s=m_.model_.spec_;
        m_.counters_.D2D_bytes+=size_t(48)*(s.ssm_state*s.ssm_state*s.ssm_rank+3*s.ssm_state*(2*s.ssm_group+s.ssm_rank))*4;
    }
    target_[p][index].launch(stream);parity(p^1);
    if(preserve)++m_.counters_.graph_verify;else ++m_.counters_.graph_replay;
}
void MtpGraphs::rewind(cudaStream_t stream){
    if(mtp_abort::selected(m_,stream))throw std::logic_error("mtp_abort discarded graph request cannot rewind");
    unsigned p=2;for(const auto &s:m_.saved_)if(s.state){p=s.parity;break;}
    if(p>1)throw std::runtime_error("MTP saved history owner missing");
    rewind_[p].launch(stream);parity(p);++m_.counters_.graph_rewind;
    const auto &s=m_.model_.spec_;
    m_.counters_.D2D_bytes+=size_t(48)*(s.ssm_state*s.ssm_state*s.ssm_rank+3*s.ssm_state*(2*s.ssm_group+s.ssm_rank))*4;
}
void MtpGraphs::commit(int count,cudaStream_t stream){
    if(count<1||count>3)throw std::invalid_argument("MTP graph commit span");commit_[count-1].launch(stream);++m_.counters_.graph_commit;
}
void MtpGraphs::restore_prefix(int count,cudaStream_t stream){
    if(!m_.retain_prefix_||count<1||count>2)throw std::invalid_argument("MTP retained prefix span");
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
