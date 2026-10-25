#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "strata/qwen35moe/gdn_dense_launch.hpp"
#include "moe_graph_state.hpp"
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
namespace strata::qwen35moe {
bool Runtime::gdn_dense_graph_requested()const{
    const char* raw=std::getenv("STRATA_QWEN35_GDN_DENSE_GRAPH");
    if(!raw||std::string(raw)=="0")return false;
    if(std::string(raw)!="1")throw std::invalid_argument("GDN_DENSE_GRAPH requires0 or1");
    if(!nvfp4_||!nvfp4_->device_tg||!tg_dynamic_||session_capacity()!=1||mtp_enabled_||
       sessions_||paired_session_||checkpoint_sink_||mtp_sink_||verify_state_||verify_history_)
        throw std::invalid_argument("GDN_DENSE_GRAPH requires ordinary NVFP4 single-session dynamic TG");
    return true;
}
void Runtime::prepare_gdn_dense_graphs(MoeGraphState& graphs){
    if(!gdn_dense_graph_requested())return;
    size_t recurrent=0,nodes=0;
    for(size_t il=0;il<spec_.layers.size();++il)if(spec_.layers[il].recurrent){
        auto& layer=layers_[il];graphs.dense_recurrent[il].capture(stream_,[&]{
            launch_gdn_dense(layer,work_,float(spec_.rms_epsilon),prefill_config_.gdn_warp_norm,stream_,
                [&](const GpuWeight& w,const void* x,float* y,int k,int n){project(w,x,y,k,n);});
        });++recurrent;nodes+=graphs.dense_recurrent[il].nodes();
    }
    if(!recurrent)throw std::runtime_error("GDN_DENSE_GRAPH model has no recurrent layers");
    graphs.dense_recurrent_enabled=true;
    std::fprintf(stderr,"GDN_DENSE_GRAPH_READY {\"layers\":%zu,\"kernel_nodes\":%zu,\"host_submissions_per_token\":%zu,\"scope\":\"RMS-GDN-residual-postRMS\"}\n",recurrent,nodes,recurrent);
}
bool Runtime::launch_gdn_dense_graph(size_t il){
    if(!decode_phase_||!moe_graph_||!moe_graph_->dense_recurrent_enabled||!nvfp4_||!nvfp4_->device_tg||
       !tg_dynamic_||sessions_||paired_session_||mtp_enabled_||checkpoint_sink_||mtp_sink_||verify_state_||verify_history_)return false;
    auto& graph=moe_graph_->dense_recurrent.at(il);
    if(!graph.ready())throw std::runtime_error("unprepared recurrent dense graph");
    graph.launch(stream_);++moe_graph_->recurrent_submissions;moe_graph_->recurrent_kernel_nodes+=graph.nodes();return true;
}
}
