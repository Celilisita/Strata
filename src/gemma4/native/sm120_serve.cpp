#include "strata/gemma4/native/sm120_api.hpp"
#include "strata/gemma4/native/sm120_options.hpp"
#include "strata/gemma4/native/sm120_budget.hpp"
#include "strata/models/request_sampling.hpp"
#include "sm120_sampled_mtp.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
namespace strata::gemma4::native {
namespace {
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
struct Stream {
    cudaStream_t value=nullptr;
    Stream(){primitive::check_cuda(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"Gemma native stream");}
    ~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}
};
std::vector<int> parse_ids(const std::string& csv){
    if(csv.empty()||csv.back()==',')throw std::invalid_argument("Gemma empty token list");
    std::vector<int> ids;std::istringstream input(csv);std::string part;
    while(std::getline(input,part,',')){
        std::size_t used=0;const int id=std::stoi(part,&used);
        if(used!=part.size()||id<0||id>=262144)throw std::invalid_argument("Gemma token outside vocabulary");
        ids.push_back(id);
    }
    return ids;
}
bool eog(int token){return token==1||token==50||token==106;}
class Service {
    Sm120Options options_;
    GgufFile file_;
    Stream stream_;
    primitive::DeviceStorage storage_;
    std::unique_ptr<strata::kernels::RadixSampler> sampler_;
    std::unique_ptr<Sm120Model> model_;
    std::unique_ptr<GgufFile> draft_file_;
    std::unique_ptr<mtp_assistant::Draft> draft_;
    std::unique_ptr<sampled_mtp::Controller> controller_;
    strata::models::RequestSampling sampling_;
    int* history_=nullptr;
    bool active_=false;
    int pending_=-1,remaining_=0;
    int pick(){
        const auto bytes=sampling_.history.size()*sizeof(int);
        if(bytes)primitive::check_cuda(cudaMemcpyAsync(history_,sampling_.history.data(),bytes,cudaMemcpyHostToDevice,stream_.value),"Gemma penalty history upload");
        sampler_->sample(model_->logits(),history_,int(sampling_.history.size()),sampling_.params,stream_.value,
            reinterpret_cast<const int*>(model_->invalid_state()));
        sampler_->write_pick(model_->next_input(),model_->invalid_state(),stream_.value);
        strata::kernels::SamplePick sample;int position=-1;
        primitive::check_cuda(cudaMemcpyAsync(&sample,sampler_->result(),sizeof(sample),cudaMemcpyDeviceToHost,stream_.value),"Gemma native sampled response");
        primitive::check_cuda(cudaMemcpyAsync(&position,model_->device_position(),sizeof(position),cudaMemcpyDeviceToHost,stream_.value),"Gemma native cursor response");
        primitive::check_cuda(cudaStreamSynchronize(stream_.value),"Gemma native token completion");
        if(sample.first_bad!=-1||sample.id<0||sample.id>=262144||position!=model_->position())
            throw std::runtime_error("Gemma native numerical/cursor/sampler state");
        ++sampling_.params.counter;return sample.id;
    }
public:
    explicit Service(Sm120Options options):options_(std::move(options)),file_(options_.model){
        // Construct the shared sampler first: its exact CUB scratch is already
        // reflected in cudaMemGetInfo before model/request budget admission.
        sampler_=std::make_unique<strata::kernels::RadixSampler>(262144);
        history_=static_cast<int*>(storage_.allocate(std::size_t(options_.context)*sizeof(int)));
        model_=make_sm120_gemma(file_,options_.batch,stream_.value);
        const auto budget=estimate_sm120_budget(file_,model_->spec(),options_.batch);
        if(model_->resident_bytes()!=budget.static_owned)throw std::logic_error("Gemma owned budget/source mismatch");
        if(options_.mtp){
            model_->prepare_mtp(options_.mtp);
            const auto scratch=model_->mtp_scratch();
            draft_file_=std::make_unique<GgufFile>(options_.draft);
            draft_=std::make_unique<mtp_assistant::Draft>(*draft_file_,stream_.value,scratch.decoded,scratch.decoded_bytes);
            draft_->select_head_backend(mtp_assistant::HeadBackend::Direct399);
            controller_=std::make_unique<sampled_mtp::Controller>(*model_,*draft_,*sampler_);
            admit_sm120_budget(0,"sampled MTP complete cold owned storage and vendor pools");
        }
        sampling_.params.temperature=0;
        std::cerr<<"Gemma candidate UNQUALIFIED backend="<<gemma_weight_format_name(gemma_weight_format_requested())<<" max_batch="<<options_.batch<<" resident_bytes="<<budget.static_owned
                 <<" vendor_pool_bound="<<budget.vendor_pool_bound<<" sampler_bytes="<<sampler_->workspace_bytes()<<" logical_context="<<options_.context<<" MTP="<<options_.mtp
                 <<" draft_metadata_context="<<(draft_?draft_->metadata_context():0)<<" MTP_mode=fixed_per_process"
                 <<" MTP_target_extra_bytes="<<(model_->resident_bytes()-budget.static_owned)
                 <<" MTP_draft_weights_bytes="<<(draft_?draft_->weight_bytes():0)
                 <<" MTP_draft_owned_scratch_bytes="<<(draft_?draft_->scratch_bytes():0)
                 <<" MTP_controller_owned_bytes="<<(controller_?controller_->bytes():0)
                 <<" MTP_borrowed_decode_bytes="<<(draft_?draft_->decoded_capacity_bytes():0)<<'\n';
    }
    void release(){
        if(controller_)controller_->release_graphs();
        model_->release_request();active_=false;pending_=-1;remaining_=0;
        sampling_={};sampling_.params.temperature=0;
    }
    void command(const std::string& line){
        std::istringstream input(line);std::string command,csv,extra;int slot=-1;
        if(!(input>>command>>slot)||slot!=0)throw std::invalid_argument("Gemma native requires slot0");
        if(command=="RELEASE"){
            if(input>>extra)throw std::invalid_argument("Gemma RELEASE fields");
            const auto counter=sampling_.params.counter;
            const auto stats=(controller_&&active_)?controller_->stats():sampled_mtp::Stats{};
            release();std::cout<<"RELEASED slot=0 mtp="<<options_.mtp<<" sampler_counter="<<counter
                <<" mtp_rounds="<<stats.rounds<<" mtp_emitted="<<stats.emitted<<" mtp_accepted="<<stats.accepted
                <<" mtp_eager_rounds="<<stats.eager_rounds<<" graph_captures="<<(stats.captures[0]+stats.captures[1]+stats.captures[2])
                <<" graph_replays="<<(stats.replays[0]+stats.replays[1]+stats.replays[2])<<'\n';return;
        }
        if(command=="CONFIG"){
            if(active_)throw std::invalid_argument("Gemma slot is active");
            if(controller_)controller_->release_graphs();
            sampling_.configure(input,options_.context);std::cout<<"CONFIGURED slot=0\n";return;
        }
        if(command=="PREPARE"){
            int maximum_new=0;
            if(!(input>>maximum_new>>csv)||input>>extra||active_||maximum_new<1)
                throw std::invalid_argument("Gemma PREPARE fields/slot");
            const auto prompt=parse_ids(csv);
            if(prompt.empty()||std::size_t(maximum_new)>std::size_t(options_.context)||
               prompt.size()>std::size_t(options_.context)-std::size_t(maximum_new))
                throw std::invalid_argument("Gemma prompt plus output exceeds logical context");
            const auto start=Clock::now();
            if(controller_)controller_->release_graphs();
            model_->reset(int(prompt.size())+maximum_new);
            const bool original=gemma_weight_format_requested()==GemmaWeightFormat::OriginalQuant;
            std::size_t free_before=0,total_before=0;
            if(original)primitive::check_cuda(cudaMemGetInfo(&free_before,&total_before),"Gemma original before actual PP");
            sampling_.prepare(prompt);model_->enqueue_prefill(prompt);pending_=pick();
            const auto setup=Clock::now();
            if(controller_)controller_->bind(sampling_.params,options_.mtp,{1,50,106},options_.graph);
            const double setup_ms=controller_?milliseconds(setup):0;
            if(original){
                // pick already joined the real PP. No dummy projection or extra synchronization.
                admit_sm120_budget(0,"original completed PP including vendor pools");
                std::size_t free_after=0,total_after=0;
                primitive::check_cuda(cudaMemGetInfo(&free_after,&total_after),"Gemma original after actual PP");
                std::cerr<<"Gemma original actualPP free_before="<<free_before<<" free_after="<<free_after
                         <<" total="<<total_after<<" reserve="<<primitive::DeviceStorage::reserve_bytes()
                         <<" observed_new_allocation="<<(free_before>free_after?free_before-free_after:0)<<'\n';
            }
            remaining_=maximum_new-1;active_=true;
            std::cout<<"PREFILL anchor="<<pending_<<" tokens="<<prompt.size()<<" pp_ms="<<milliseconds(start)
                     <<" speculative="<<int(bool(controller_))<<" mtp="<<options_.mtp<<" sampler_counter="<<sampling_.params.counter
                     <<" pp_evaluated="<<prompt.size()<<" pp_reused=0 mtp_setup_ms="<<setup_ms<<" graph_capture_ms=0"
                     <<" kv_bytes="<<model_->kv_bytes()<<" max_batch="<<options_.batch<<" aux=0 cache_slots=0\n";return;
        }
        if(command=="STEP"){
            int anchor=-1,requested=0;
            if(!(input>>anchor>>requested)||input>>extra||!active_||anchor!=pending_||eog(anchor)||
               requested<1||requested>remaining_||model_->position()>=model_->capacity())
                throw std::invalid_argument("Gemma STEP pending/remaining/context");
            const auto start=Clock::now();sampling_.remember(anchor);
            if(controller_){
                const int before=model_->position();
                const auto result=controller_->round(sampling_.history,sampling_.params.counter,requested);
                if(result.invalid||result.count<1||result.count>(std::min)(requested,options_.mtp+1)||
                   result.accepted!=result.count-1||result.proposed<result.accepted||result.proposed>options_.mtp||
                   result.position!=before+result.count||model_->position()!=result.position)
                    throw std::runtime_error("Gemma sampled MTP response/count/cursor contract");
                for(int i=0;i<result.count;++i){
                    if(result.ids[i]<0||result.ids[i]>=262144||(i+1<result.count&&eog(result.ids[i])))
                        throw std::runtime_error("Gemma sampled MTP token/EOG extent");
                }
                // Entered this round with pending remembered exactly once. Only
                // emitted IDs before the new pending join host penalty history.
                for(int i=0;i+1<result.count;++i)sampling_.remember(result.ids[i]);
                sampling_.params.counter+=std::uint64_t(result.count);
                pending_=result.ids[result.count-1];remaining_-=result.count;
                std::cout<<"STEPPED next="<<pending_<<" ms="<<milliseconds(start)
                         <<" proposed="<<result.proposed<<" accepted="<<result.accepted
                         <<" matched="<<result.matched<<" finished="<<result.finished<<" reason="<<result.reason
                         <<" position="<<result.position<<" sampler_counter="<<sampling_.params.counter<<" ids=";
                for(int i=0;i<result.count;++i)std::cout<<(i?",":"")<<result.ids[i];
                const auto& timing=controller_->lastTiming();
                if(timing.enabled)std::cout<<" timing_enabled=1 phase_draft_gpu_ms="<<timing.draft_gpu_ms
                    <<" phase_target_gpu_ms="<<timing.target_gpu_ms<<" phase_sample_gpu_ms="<<timing.sample_gpu_ms
                    <<" phase_commit_gpu_ms="<<timing.commit_gpu_ms<<" phase_total_gpu_ms="<<timing.total_gpu_ms;
                std::cout<<'\n';return;
            }
            model_->enqueue_step(model_->next_input(),options_.graph);pending_=pick();--remaining_;
            std::cout<<"STEPPED next="<<pending_<<" ms="<<milliseconds(start)<<" proposed=0 accepted=0 position="<<model_->position()<<" sampler_counter="<<sampling_.params.counter<<" ids="<<pending_<<'\n';return;
        }
        throw std::invalid_argument("unsupported Gemma native command");
    }
};
}
int gemma4_sm120_serve(int argc,char** argv){
    try{
        const auto options=Sm120Options::parse(argc,argv);Service service(options);
        std::cout<<"READY slots=1\n"<<std::flush;std::string line;
        while(std::getline(std::cin,line)){
            if(line=="QUIT")break;
            try{if(line.size()>std::size_t(options.context)*24+1024)throw std::invalid_argument("Gemma wire command too long");
                service.command(line);
            }catch(const std::exception& error){service.release();std::cout<<"ERROR "<<error.what()<<'\n';}
            std::cout<<std::flush;
        }
        return 0;
    }catch(const std::exception& error){std::cerr<<"Gemma native service: "<<error.what()<<'\n';return 1;}
}
}
