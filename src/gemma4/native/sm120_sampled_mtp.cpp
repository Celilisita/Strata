#include "sm120_sampled_mtp.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
namespace strata::gemma4::native::sampled_mtp {
namespace {
void require(bool yes,const char* what){if(!yes)throw std::invalid_argument(what);}
void checked(cudaError_t status,const char* what){primitive::check_cuda(status,what);}
bool phase_timing_requested(){
    const char* p=std::getenv("STRATA_GEMMA_MTP_PHASE_TIMING");
    if(!p)return false;
    require((p[0]=='0'||p[0]=='1')&&p[1]=='\0',"MTP phase timing flag must be0or1");
    return p[0]=='1';
}
CacheView view(const Sm120MtpCacheLease& c){return {const_cast<void*>(c.key),const_cast<void*>(c.value),c.capacity,c.heads,c.dim,c.window};}
bool same(const Sm120MtpCacheLease& a,const Sm120MtpCacheLease& b){
    return a.owner==b.owner&&a.epoch==b.epoch&&a.key==b.key&&a.value==b.value&&
        a.capacity==b.capacity&&a.heads==b.heads&&a.dim==b.dim&&a.window==b.window;
}
}
Controller::Controller(Sm120Model& main,mtp_assistant::Draft& draft,kernels::RadixSampler& sampler)
    :main_(main),draft_(draft),sampler_(sampler),stream_(main.ordered_stream()){
    const auto scratch=main.mtp_scratch();
    require(scratch.owner==&main&&scratch.stream==stream_&&scratch.decoded&&scratch.decoded_bytes>=128u*1024*1024&&
        draft.borrows_decode()&&draft.decoded_capacity_bytes()==128u*1024*1024&&draft.resident_tensors().size()==49,
        "sampled MTP one borrowed49-tensor Assistant/stream");
    try{
        state_=static_cast<Round*>(memory_.allocate(sizeof(Round)));eog_=static_cast<int*>(memory_.allocate(16*4));
        const auto h=std::size_t(main.spec().context);require(h>=1&&h<=262144,"sampled MTP history bound");
        history_=static_cast<int*>(memory_.allocate(h*4));rows_history_=static_cast<int*>(memory_.allocate(3*h*4));
        if(phase_timing_requested())phase_ticks_=static_cast<std::uint64_t*>(memory_.allocate(5*sizeof(std::uint64_t)));
        last_timing_.enabled=phase_ticks_!=nullptr;
        checked(cudaMemsetAsync(state_,0,sizeof(Round),stream_),"sampled MTP cold metadata");
        checked(cudaStreamSynchronize(stream_),"sampled MTP cold allocation lease");
    }catch(...){cudaStreamSynchronize(stream_);throw;}
}
Controller::~Controller(){if(stream_)cudaStreamSynchronize(stream_);destroy_graphs();}
void Controller::destroy_graphs()noexcept{for(int i=0;i<3;++i){if(exec_[i])cudaGraphExecDestroy(exec_[i]);if(graphs_[i])cudaGraphDestroy(graphs_[i]);exec_[i]=nullptr;graphs_[i]=nullptr;}}
void Controller::release_graphs(){checked(cudaStreamSynchronize(stream_),"sampled MTP reset/graph join");destroy_graphs();bound_=false;eager_seen_=false;last_timing_={};last_timing_.enabled=phase_ticks_!=nullptr;}
void Controller::bind(const kernels::SamplerParams& p,int requested,const std::vector<int>& eog,bool graph_enabled){
    require(!bound_&&std::none_of(exec_.begin(),exec_.end(),[](auto x){return x!=nullptr;}),"release sampled MTP graphs before bind");
    kernels::RadixSampler::validate(p);require((requested==1||requested==2)&&requested<=main_.mtp_drafts()&&eog.size()<=16&&p.penalty_last_n<=main_.spec().context,"sampled MTP request policy");
    const auto lease=main_.mtp_lease();const auto local=main_.mtp_cache(58),global=main_.mtp_cache(59);
    require(lease.owner==&main_&&lease.epoch&&lease.stream==stream_&&lease.seed&&lease.position&&lease.pending&&lease.invalid&&
        lease.decoded&&lease.decoded_bytes>=128u*1024*1024&&local.owner==lease.owner&&global.owner==lease.owner&&
        local.epoch==lease.epoch&&global.epoch==lease.epoch&&local.key&&local.value&&global.key&&global.value&&
        local.heads==16&&local.dim==256&&local.window==1024&&global.heads==4&&global.dim==512&&!global.window&&
        global.capacity==main_.capacity()&&local.capacity>=std::min(main_.capacity(),1025+requested)&&main_.position()>0,
        "sampled MTP processed Main58/59 lease");
    require(draft_.head_backend()==mtp_assistant::HeadBackend::Direct399,"sampled MTP explicit direct originalQ4 Assistant head");
    for(int id:eog)require(id>=0&&id<262144,"sampled MTP EOG range");
    require(!poisoned_||lease.epoch!=lease_.epoch,"sampled MTP failed request needs fresh reset epoch");
    std::array<int,16> eog_host{};std::copy(eog.begin(),eog.end(),eog_host.begin());
    checked(cudaMemcpyAsync(eog_,eog_host.data(),sizeof(eog_host),cudaMemcpyHostToDevice,stream_),"sampled MTP cold EOG policy");
    checked(cudaStreamSynchronize(stream_),"sampled MTP cold policy lease");
    params_=p;params_.counter=0;requested_=requested;eog_count_=int(eog.size());
    const bool penalized=p.penalty_repeat!=1||p.penalty_freq!=0||p.penalty_present!=0;
    history_length_=penalized?p.penalty_last_n:0;lease_=lease;local_=local;global_=global;
    head_epoch_=draft_.head_policy_epoch();graph_enabled_=graph_enabled;bound_=true;poisoned_=false;stats_={};
}
void Controller::validate()const{
    require(bound_&&!poisoned_&&draft_.head_policy_epoch()==head_epoch_,"sampled MTP stale/failed binding");
    const auto now=main_.mtp_lease();
    require(now.owner==lease_.owner&&now.epoch==lease_.epoch&&now.stream==stream_&&now.seed==lease_.seed&&
        now.position==lease_.position&&now.pending==lease_.pending&&now.invalid==lease_.invalid&&
        now.decoded==lease_.decoded&&now.decoded_bytes==lease_.decoded_bytes&&same(main_.mtp_cache(58),local_)&&
        same(main_.mtp_cache(59),global_),"sampled MTP owner/epoch/arena/KV changed");
}
void Controller::submit(int drafts){
    // Both assistant unrolls read the same committed KV and same P/P-1 bounds,
    // exactly as the qualified V100 controller; no synthetic future KV exists.
    if(phase_ticks_)phase_stamp(phase_ticks_,0,stream_);
    prepare_round(state_,lease_.position,lease_.invalid,local_.capacity,global_.capacity,
        drafts?draft_.metadata_context():main_.spec().context,stream_);
    if(drafts){
        auto embedding=main_.enqueue_scaled_embedding(lease_.pending);
        draft_.run(embedding,lease_.seed,view(local_),view(global_),&state_->bounds);
        auto output=draft_.output();save_draft(state_,output.proposal,output.invalid,0,stream_);
        if(drafts==2){embedding=main_.enqueue_scaled_embedding(state_->drafts);
            draft_.run(embedding,output.post_state,view(local_),view(global_),&state_->bounds);
            output=draft_.output();save_draft(state_,output.proposal,output.invalid,1,stream_);}
    }
    form_inputs(state_,lease_.pending,drafts,stream_);
    if(phase_ticks_)phase_stamp(phase_ticks_,1,stream_);
    main_.enqueue_target_rows(state_->inputs,drafts+1);
    if(phase_ticks_)phase_stamp(phase_ticks_,2,stream_);
    const auto target=main_.target_rows();require(target.rows==drafts+1&&target.features&&target.logits,"sampled MTP target rows");
    prepare_histories(state_,history_,rows_history_,history_length_,drafts,lease_.invalid,stream_);
    for(int i=0;i<=drafts;++i){
        sampler_.sample_counter(target.logits+std::size_t(i)*262144,
            history_length_?rows_history_+std::size_t(i)*history_length_:nullptr,history_length_,params_,stream_,
            &state_->counter,unsigned(i),reinterpret_cast<const int*>(&state_->invalid));
        // The shared sampler overwrites result() on the next row. Keep each
        // target draw separately before enqueuing another sample chain.
        checked(cudaMemcpyAsync(state_->picks+i,sampler_.result(),sizeof(kernels::SamplePick),cudaMemcpyDeviceToDevice,stream_),"sampled MTP independent row pick");
    }
    decide(state_,drafts,main_.capacity(),eog_,eog_count_,lease_.invalid,stream_);
    if(phase_ticks_)phase_stamp(phase_ticks_,3,stream_);
    main_.enqueue_commit_prefix(&state_->response.count,&state_->next,&state_->invalid);
    finalize(state_,lease_.position,lease_.invalid,stream_);
    if(phase_ticks_)phase_stamp(phase_ticks_,4,stream_);
}
void Controller::capture(int drafts){
    bool active=false;try{checked(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal),"sampled MTP capture");active=true;submit(drafts);
        checked(cudaStreamEndCapture(stream_,&graphs_[drafts]),"sampled MTP capture end");active=false;
        checked(cudaGraphInstantiateWithFlags(&exec_[drafts],graphs_[drafts],0),"sampled MTP instantiate");
        checked(cudaGraphGetNodes(graphs_[drafts],nullptr,&stats_.nodes[drafts]),"sampled MTP complete graph nodes");++stats_.captures[drafts];
    }catch(...){if(active){cudaGraph_t discard=nullptr;cudaStreamEndCapture(stream_,&discard);if(discard)cudaGraphDestroy(discard);}throw;}
}
Response Controller::round(const std::vector<int>& history,std::uint64_t counter,int remaining){
    validate();const int p=main_.position(),room=main_.capacity()-p;
    require(remaining>0&&room>0&&remaining<=room&&counter<=(std::numeric_limits<std::uint64_t>::max)()-2,"sampled MTP budget/counter");
    for(int id:history)require(id>=0&&id<262144,"sampled MTP history ID");
    const int drafts=p>=draft_.metadata_context()?0:std::min({requested_,remaining-1,room-1});
    std::vector<int> staged(std::size_t(history_length_),-1);const auto take=std::min(staged.size(),history.size());
    std::copy(history.end()-take,history.end(),staged.end()-take);
    const std::uint64_t draw=counter;const int budget=remaining;Response response{};
    std::array<std::uint64_t,5> phase_host{};
    try{
        if(!staged.empty())checked(cudaMemcpyAsync(history_,staged.data(),staged.size()*4,cudaMemcpyHostToDevice,stream_),"sampled MTP actual history upload");
        checked(cudaMemcpyAsync(&state_->counter,&draw,sizeof(draw),cudaMemcpyHostToDevice,stream_),"sampled MTP absolute draw counter");
        checked(cudaMemcpyAsync(&state_->remaining,&budget,sizeof(budget),cudaMemcpyHostToDevice,stream_),"sampled MTP remaining budget");
        if(!graph_enabled_||!eager_seen_){submit(drafts);++stats_.eager_rounds;eager_seen_=true;}
        else{if(!exec_[drafts])capture(drafts);checked(cudaGraphLaunch(exec_[drafts],stream_),"sampled MTP complete round replay");++stats_.replays[drafts];}
        checked(cudaMemcpyAsync(&response,&state_->response,sizeof(response),cudaMemcpyDeviceToHost,stream_),"sampled MTP one response");
        if(phase_ticks_)checked(cudaMemcpyAsync(phase_host.data(),phase_ticks_,sizeof(phase_host),cudaMemcpyDeviceToHost,stream_),"sampled MTP phase ticks40B");
        checked(cudaStreamSynchronize(stream_),"sampled MTP round/host source lease");
        if(phase_ticks_){
            require(phase_host[0]!=0&&std::is_sorted(phase_host.begin(),phase_host.end()),"MTP phase globaltimer nonmonotonic");
            last_timing_.ticks=phase_host;
            last_timing_.draft_gpu_ms=double(phase_host[1]-phase_host[0])*1e-6;
            last_timing_.target_gpu_ms=double(phase_host[2]-phase_host[1])*1e-6;
            last_timing_.sample_gpu_ms=double(phase_host[3]-phase_host[2])*1e-6;
            last_timing_.commit_gpu_ms=double(phase_host[4]-phase_host[3])*1e-6;
            last_timing_.total_gpu_ms=double(phase_host[4]-phase_host[0])*1e-6;
        }
        require(!response.invalid&&response.count>=1&&response.count<=drafts+1&&response.count<=remaining&&
            response.accepted==response.count-1&&response.proposed==drafts&&response.matched>=response.accepted&&
            response.matched<=drafts&&response.position==p+response.count&&response.reason!=int(Stop::Invalid),"sampled MTP invalid commit response");
        for(int i=0;i<3;++i)require(i<response.count?(response.ids[i]>=0&&response.ids[i]<262144):response.ids[i]==0,"sampled MTP output ID/padding");
        require(response.finished==(response.reason!=int(Stop::Continue))&&
            (response.reason!=int(Stop::Budget)||response.count==remaining),"sampled MTP stop/budget response");
        main_.finish_target_commit(response.count,response.position,lease_.epoch);
        ++stats_.rounds;stats_.emitted+=response.count;stats_.accepted+=response.accepted;stats_.proposed+=response.proposed;
        stats_.history_upload_bytes+=staged.size()*4;stats_.metadata_upload_bytes+=sizeof(draw)+sizeof(budget);return response;
    }catch(...){cudaStreamSynchronize(stream_);poisoned_=true;throw;}
}
}
