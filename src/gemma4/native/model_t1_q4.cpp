#include "strata/gemma4/native/model.hpp"
namespace strata::gemma4::native {
void Model::select_t1_q4_ffn(T1Q4FfnPolicy policy){
    if(policy!=T1Q4FfnPolicy::Off&&policy!=T1Q4FfnPolicy::GateUp&&policy!=T1Q4FfnPolicy::Both)
        throw std::invalid_argument("Gemma T1 Q4 FFN request policy");
    if(host_position_!=0)throw std::invalid_argument("Gemma T1 Q4 policy requires reset P0");
    if(ffn_->t1_q4_ffn_policy()==policy)return;
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma cold T1 Q4 policy drain");
    release_graph();
    ffn_->select_t1_q4_ffn(policy);
    ++reset_epoch_;
}
}
