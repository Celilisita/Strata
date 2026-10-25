#include "strata/gemma4/native/model.hpp"
namespace strata::gemma4::native {
void Model::enqueue_base_tail(VerifyPolicy policy){
    if(!capacity_||host_position_<1||host_position_>=capacity_)
        throw std::invalid_argument("Gemma base tail requires processed prefix and context room");
    validate_verify_policy(policy,1);
    // Graph replay reads the current device cursor, pending and budget. The
    // ordinary T1 owns model math, seed, capped pick and cursor advance.
    detail::begin_base_tail(verify_packet_,position_,next_,capacity_,spec_.vocabulary,policy,stream_);
    forward(next_,1,true);
    detail::finish_base_tail(verify_packet_,position_,next_,capacity_,spec_.vocabulary,policy,stream_);
}
}
