#include "strata/qwen35/completed_v100/native/mtp.hpp"
#include "strata/qwen35/completed_v100/native/mtp_draft_link.hpp"
#include "mtp_graph.hpp"
#include "mtp_abort.hpp"
namespace strata::completed_v100::qwen35::native {
std::vector<int> NativeMtp::round_gpu_chain(int known,int position,int proposals,cudaStream_t stream){
    if(graph_enabled())graphs_->begin(position,stream);
    host_->p->known=known;
    check_cuda(cudaMemcpyAsync(device_ids_,&host_->p->known,4,cudaMemcpyHostToDevice,stream),"MTP one known GPU-chain ID");
    counters_.H2D_bytes+=4;
    const int h=int(model_.spec_.n_embd);
    for(int t=0;t<proposals;++t){
        if(graph_enabled()){
            graphs_->draft(t,stream);counters_.D2D_bytes+=size_t(2*h)*4;
        }else{
            enqueue_draft(device_ids_+t,t?draft_seed_:seed_,position+t,stream);
            mtp_draft_link(picked_,invalid_,int(model_.spec_.n_vocab),device_ids_+t+1,
                           draft_ids_+t,draft_flags_+t,stream);
        }
        counters_.D2D_bytes+=12;
    }
    if(proposals&&!graph_enabled())snapshot(stream);
    std::vector<int> predicted;
    try{
        predicted=target(device_ids_,proposals+1,position,stream,proposals!=0,proposals!=0);
        if(mtp_abort::take_fault(*this))throw std::runtime_error("mtp_abort injected post-target discard");
        for(int t=0;t<proposals;++t)
            if(host_->p->draft_flags[t]||host_->p->draft_ids[t]<0||host_->p->draft_ids[t]>=int(model_.spec_.n_vocab))
                throw std::runtime_error("invalid GPU-chained draft logits");
    }catch(...){
        // Invalid drafts use safe embedding placeholders on device. Restore
        // the exact target transaction before exposing the error on host.
        if(mtp_abort::selected(*this,stream))mtp_abort::poison(*this);
        else if(proposals){if(graph_enabled())graphs_->rewind(stream);else rewind(stream);}
        throw;
    }
    int accepted=0;
    while(accepted<proposals&&host_->p->draft_ids[accepted]==predicted[accepted])++accepted;
    if(accepted<proposals){
        reject_prefix(accepted+1,predicted,position,stream);
    }
    const int consumed=accepted+1;
    if(graph_enabled()){graphs_->commit(consumed,stream);counters_.D2D_bytes+=size_t(2*consumed)*h*4;}
    else enqueue_commit(consumed,position,stream);
    if(sampling_)counters_.D2D_bytes+=sampling_->commit(device_ids_,consumed,position,stream);
    counters_.D2D_bytes+=size_t(consumed+1)*h*4;
    ++counters_.rounds;counters_.proposed+=proposals;counters_.accepted+=accepted;
    return predicted;
}
}
