#include "strata/gemma4/native/draft_round.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace strata::gemma4::native {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point first){return std::chrono::duration<double>(Clock::now()-first).count();}
void checked(cudaError_t status,const char* where){primitive::check_cuda(status,where);}
bool same(const MainCacheLease& a,const MainCacheLease& b){
    return a.owner==b.owner&&a.epoch==b.epoch&&a.key==b.key&&a.value==b.value&&
        a.capacity==b.capacity&&a.heads==b.heads&&a.dim==b.dim&&a.window==b.window;
}
CacheView view(const MainCacheLease& c){
    // The205 API's CacheView has mutable pointer types, but its assistant
    // kernels only read these payloads; controller never writes either cache.
    return {const_cast<void*>(c.key),const_cast<void*>(c.value),c.capacity,c.heads,c.dim,c.window};
}
}
DraftRoundController::DraftRoundController(Model& main,assistant::Draft& draft):main_(main),draft_(draft){
    const auto scratch=main_.draft_scratch();
    if(!scratch.owner||!scratch.stream||!scratch.decoded||scratch.decoded_bytes<128u*1024*1024)
        throw std::invalid_argument("Gemma218 stable main scratch lease");
    owner_=scratch.owner;stream_=scratch.stream;decoded_=scratch.decoded;decoded_bytes_=scratch.decoded_bytes;
    // Keep the131072 GGUF metadata. The new history profile separately
    // selects its262144 runtime position range; no frequency payload changes.
    assistant_context_=main_.global_history_enabled()?main_.spec().context:draft_.metadata_context();
    draft_.select_runtime_context(assistant_context_);
    state_=static_cast<detail::DraftRoundState*>(memory_.allocate(sizeof(detail::DraftRoundState)));
    stats_.assistant_weight_bytes=draft_.weight_bytes();stats_.assistant_scratch_bytes=draft_.scratch_bytes();
    stats_.borrowed_decode_bytes=draft_.decoded_capacity_bytes();stats_.metadata_bytes=memory_.bytes();
    stats_.assistant_tensors=draft_.resident_tensors().size();
    if(!draft_.borrows_decode()||stats_.assistant_tensors!=49||stats_.borrowed_decode_bytes!=128u*1024*1024)
        throw std::runtime_error("Gemma218 one49-tensor borrowed-decode Draft required");
}
DraftRoundController::~DraftRoundController(){
    if(stream_)cudaStreamSynchronize(stream_);
    destroy_graphs();
}
void DraftRoundController::destroy_graphs()noexcept{
    for(int i=0;i<3;++i){
        if(executables_[i])cudaGraphExecDestroy(executables_[i]);
        if(graphs_[i])cudaGraphDestroy(graphs_[i]);
        executables_[i]=nullptr;graphs_[i]=nullptr;
    }
}
void DraftRoundController::release_graphs(){
    checked(cudaStreamSynchronize(stream_),"Gemma218 cold graph lifetime join");
    destroy_graphs();bound_=false;finished_=true;remaining_=0;epoch_=0;
}
void DraftRoundController::validate_binding()const{
    if(!bound_)throw std::invalid_argument("Gemma218 request not bound");
    if(draft_.head_policy_epoch()!=draft_head_epoch_)
        throw std::invalid_argument("401 stale Draft head policy; release Graphs and rebind");
    const auto scratch=main_.draft_scratch();const auto current=main_.draft_lease();
    if(scratch.owner!=owner_||scratch.stream!=stream_||scratch.decoded!=decoded_||scratch.decoded_bytes!=decoded_bytes_||
       current.owner!=owner_||current.epoch!=epoch_||current.stream!=stream_||
       current.seed!=main_lease_.seed||current.position!=main_lease_.position||current.pending!=main_lease_.pending||
       current.invalid!=main_lease_.invalid||current.decoded!=decoded_||current.decoded_bytes!=decoded_bytes_||
       main_.position()!=position_||!same(main_.draft_cache(58),local_)||!same(main_.draft_cache(59),global_))
        throw std::invalid_argument("Gemma218 stale owner/epoch/cache/cursor; release before reset or policy change");
}
void DraftRoundController::bind(int remaining,const std::vector<int>& eog,int requested,int capacity){
    const auto first=Clock::now();
    if(bound_||std::any_of(executables_.begin(),executables_.end(),[](auto p){return p!=nullptr;})||
       remaining<0||(requested!=1&&requested!=2)||eog.size()>16)
        throw std::invalid_argument("Gemma218 release graphs before binding request");
    const auto scratch=main_.draft_scratch();const auto current=main_.draft_lease();
    const auto local=main_.draft_cache(58),global=main_.draft_cache(59);const int p=main_.position();
    if(scratch.owner!=owner_||scratch.stream!=stream_||scratch.decoded!=decoded_||scratch.decoded_bytes!=decoded_bytes_||
       current.owner!=owner_||current.stream!=stream_||current.decoded!=decoded_||current.decoded_bytes!=decoded_bytes_||
       !current.epoch||!current.seed||!current.position||!current.pending||!current.invalid||
       local.owner!=owner_||global.owner!=owner_||local.epoch!=current.epoch||global.epoch!=current.epoch||
       !local.key||!local.value||!global.key||!global.value||local.heads!=16||local.dim!=256||local.window!=1024||
       global.heads!=4||global.dim!=512||global.window!=0||global.capacity!=capacity||
       capacity<1||capacity>main_.spec().context||p<1||p>capacity||remaining>capacity-p||
       local.capacity<(std::min)(capacity,1025+requested)||(poisoned_&&current.epoch==failed_epoch_))
        throw std::invalid_argument("Gemma218 processed prefix/context/cache/reset lease");
    detail::DraftRoundState initial{};initial.remaining=remaining;
    for(size_t i=0;i<eog.size();++i){
        if(eog[i]<0||eog[i]>=main_.spec().vocabulary)throw std::invalid_argument("Gemma218 EOG ID range");
        initial.eog_ids[i]=eog[i];
    }
    checked(cudaMemcpyAsync(state_,&initial,sizeof(initial),cudaMemcpyHostToDevice,stream_),"Gemma218 cold policy metadata");
    checked(cudaStreamSynchronize(stream_),"Gemma218 cold policy upload lifetime");
    DraftRoundStats reset{};
    reset.assistant_construct_wall_s=stats_.assistant_construct_wall_s;
    reset.assistant_weight_bytes=stats_.assistant_weight_bytes;reset.assistant_scratch_bytes=stats_.assistant_scratch_bytes;
    reset.borrowed_decode_bytes=stats_.borrowed_decode_bytes;reset.metadata_bytes=stats_.metadata_bytes;
    reset.assistant_tensors=stats_.assistant_tensors;reset.bind_H2D_bytes=sizeof(initial);reset.bind_wall_s=elapsed(first);
    stats_=reset;main_lease_=current;local_=local;global_=global;epoch_=current.epoch;
    const auto head_epoch=draft_.head_policy_epoch();
    if(head_epoch!=draft_head_epoch_)draft_warmed_=false;
    draft_head_epoch_=head_epoch;
    position_=p;capacity_=capacity;remaining_=remaining;requested_=requested;eog_count_=int(eog.size());
    bound_=true;finished_=remaining==0;poisoned_=false;
}
void DraftRoundController::enqueue(int drafts){
    const VerifyPolicy policy{&state_->remaining,eog_count_?state_->eog_ids:nullptr,eog_count_};
    if(!drafts){main_.enqueue_base_tail(policy);return;}
    detail::prepare_draft_round(main_lease_.position,main_lease_.invalid,state_,
        local_.capacity,global_.capacity,assistant_context_,stream_);
    const float* embedding=main_.enqueue_scaled_embedding(main_lease_.pending);
    draft_.run(embedding,main_lease_.seed,view(local_),view(global_),&state_->bounds,main_.global_history_view(59));
    auto output=draft_.output();
    detail::save_draft_round(output.proposal,output.invalid,state_,0,stream_);
    if(drafts==2){
        embedding=main_.enqueue_scaled_embedding(state_->drafts);
        draft_.run(embedding,output.post_state,view(local_),view(global_),&state_->bounds,main_.global_history_view(59));
        output=draft_.output();
        // First flag was already aggregated before Draft2 cleared its own.
        detail::save_draft_round(output.proposal,output.invalid,state_,1,stream_);
    }
    main_.enqueue_verify(state_->drafts,state_->drafts+1,drafts,policy,&state_->invalid);
}
void DraftRoundController::capture(int drafts){
    const auto first=Clock::now();
    if(drafts&&!draft_warmed_){
        // One read-only Draft warm prepares its cuBLAS paths. Same-stream
        // ordering makes finish_verify include this GPU work in first-round TG.
        detail::prepare_draft_round(main_lease_.position,main_lease_.invalid,state_,
            local_.capacity,global_.capacity,assistant_context_,stream_);
        const float* embedding=main_.enqueue_scaled_embedding(main_lease_.pending);
        draft_.run(embedding,main_lease_.seed,view(local_),view(global_),&state_->bounds,main_.global_history_view(59));
        const auto output=draft_.output();
        detail::save_draft_round(output.proposal,output.invalid,state_,0,stream_);
        draft_warmed_=true;++stats_.warm_draft_runs;
    }
    checked(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal),"Gemma218 round capture begin");
    try{enqueue(drafts);}
    catch(...){
        cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(stream_,&abandoned);
        if(abandoned)cudaGraphDestroy(abandoned);throw;
    }
    checked(cudaStreamEndCapture(stream_,&graphs_[drafts]),"Gemma218 round capture end");
    checked(cudaGraphInstantiateWithFlags(&executables_[drafts],graphs_[drafts],0),"Gemma218 round graph instantiate");
    checked(cudaGraphGetNodes(graphs_[drafts],nullptr,&stats_.graph_nodes[drafts]),"Gemma218 cold graph node inventory");
    ++stats_.captures[drafts];stats_.capture_host_wall_s[drafts]+=elapsed(first);
}
VerifyResponse DraftRoundController::round(){
    validate_binding();
    if(poisoned_||finished_||remaining_<1)throw std::invalid_argument("Gemma218 completed or failed round");
    const int room=capacity_-position_;
    if(room<1)throw std::invalid_argument("Gemma218 no reserved main context remains");
    const bool outside=position_<1||position_>=assistant_context_;
    const int drafts=outside?0:(std::min)({requested_,remaining_-1,room-1});
    try{
        if(!executables_[drafts])capture(drafts);
        checked(cudaGraphLaunch(executables_[drafts],stream_),"Gemma218 complete GPU draft/verify round");
        const auto response=main_.finish_verify();
        if(response.count<0||response.count>drafts+1||response.count>remaining_||
           response.position!=position_+response.count||main_.position()!=response.position)
            throw std::runtime_error("Gemma218 response prefix/budget/cursor");
        const int left=remaining_-response.count;
        if((response.reason==VerifyStop::Budget&&left!=0)||
           (response.reason==VerifyStop::Continue&&left==0))
            throw std::runtime_error("Gemma218 GPU and host remaining budget disagree");
        remaining_=left;position_=response.position;finished_=response.finished!=0;
        ++stats_.rounds;++stats_.replays[drafts];stats_.emitted+=response.count;
        stats_.accepted+=response.accepted;stats_.raw_matched+=response.matched;
        if(!drafts){++stats_.base_T1_tail_rounds;
            stats_.tail_budget+=remaining_+response.count==1;stats_.tail_room+=room==1;
            stats_.tail_assistant_context+=outside;}
        return response;
    }catch(...){poisoned_=true;failed_epoch_=epoch_;throw;}
}
}
