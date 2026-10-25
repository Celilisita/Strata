#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
// Immutable for the process: captured snapshot/rewind nodes cannot disagree after an environment change.
inline bool history_snapshot_elision_requested(){
    static const bool enabled=[](){
        const char *raw=std::getenv("STRATA_DENSE_ELIDE_HISTORY_SNAPSHOT");const std::string value=raw?raw:"0";
        if(value!="0"&&value!="1")throw std::invalid_argument("history snapshot mode must be0 or1");
        return value=="1";
    }();return enabled;
}
// MtpGraphs invokes snapshot only for its single target2/3 capture in this exact configuration.
// Eager transactions and check_rewind retain the original complete snapshot.
inline bool history_snapshot_elision_enabled(bool capture,bool gpu_chain,bool retained,int capacity){
    return history_snapshot_elision_requested()&&capture&&gpu_chain&&retained&&capacity==2;
}
inline void validate_history_snapshot_banks(const float *a,const float *b,size_t bytes){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(!a||!b||!bytes||x>std::numeric_limits<std::uintptr_t>::max()-bytes||y>std::numeric_limits<std::uintptr_t>::max()-bytes||
       (x<y+bytes&&y<x+bytes))throw std::logic_error("history snapshot requires two disjoint history banks");
}
}
