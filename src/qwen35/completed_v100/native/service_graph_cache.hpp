#pragma once
#include "strata/qwen35/completed_v100/native/mtp.hpp"
#include <string>
namespace strata::completed_v100::qwen35::native {
// One continuation-graph lease, never a prefix/KV/result cache. DenseService
// and its input/stream/model owners outlive the lease. This opt-in profile
// has one fixed executor configuration; all other profiles invalidate it.
class ServiceGraphCache {
    std::string configuration_,key_,pending_,profile_;
    bool requested_=false,hit_=false;
public:
    // Policy changes invalidate device graphs while retaining this request's CONFIG/profile.
    void invalidate(NativeMtp* mtp){
        if(mtp)mtp->release_graphs();key_.clear();pending_.clear();hit_=false;
    }
    void configure(const std::string& command){configuration_=command;}
    void select(bool requested,const std::string& profile){requested_=requested;profile_=profile;}
    bool requested()const{return requested_;}
    bool hit()const{return hit_;}
    void release(NativeMtp* mtp){
        if(mtp&&requested_&&!key_.empty()&&!mtp->prefix_graph_enabled())mtp->park_graphs();
        else{if(mtp)mtp->release_graphs();key_.clear();}
        requested_=hit_=false;pending_.clear();configuration_.clear();profile_.clear();
    }
    void prepare(NativeMtp* mtp,int tokens,int maximum,int mode,bool eligible){
        // Exact bounds ensure reserve_cache cannot grow either live arena.
        // CONFIG includes sampler mode, seed, filters and penalties. EOG mask is
        // fixed by the immutable service/model options;
        // a different prompt of equal length may use the same GPU addresses.
        pending_=requested_&&eligible&&mode==2&&!configuration_.empty()?
            profile_+":"+configuration_+":"+std::to_string(tokens)+":"+std::to_string(maximum)+":2":std::string{};
        hit_=mtp&&!pending_.empty()&&pending_==key_&&mtp->graph_nodes()>0&&!mtp->prefix_graph_enabled();
        if(hit_)mtp->park_graphs();
        else{if(mtp)mtp->release_graphs();key_.clear();}
    }
    void captured(NativeMtp* mtp){
        key_=mtp&&mtp->graph_enabled()&&!mtp->prefix_graph_enabled()?pending_:std::string{};
    }
};
}
