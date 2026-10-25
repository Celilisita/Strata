#pragma once
#include "gdn_float_value.hpp"
#include "gdn_delta_half.hpp"
#include "gdn_qk_group_policy.hpp"
#include <cstdint>
#include <cstdlib>
#include <string>
namespace strata::qwen35::native {
struct GdnPrepareResources {int registers,blocks;size_t shared_bytes,local_bytes;};
struct GdnRegisterResources {GdnPrepareResources legacy,grouped;};
GdnRegisterResources gdn_register_prepare_resources();
void gdn_register_prepare_configure();
void gdn_register_prepare(GdnInputs,GdnShape,GdnFloatValueWork,cudaStream_t);
inline uint64_t &gdn_register_submissions(){static thread_local uint64_t value=0;return value;}
inline bool gdn_register_requested(){
    const char *flag=std::getenv("STRATA_DENSE_GDN_REGISTER_COLUMN");
    if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("GDN register column mode");
    return !flag||flag[0]!='0';
}
inline void gdn_register_run(GdnInputs x,GdnShape s,float *out,float *state,void *storage,size_t bytes,cudaStream_t stream){
    if(!x.q||!x.k||!x.v||!x.log_decay||!x.beta||!x.state||!out||!state||!storage||
       s.key_dim!=128||s.value_dim!=128||s.query_heads<1||s.value_heads<1||s.value_heads>256||
       s.value_heads%s.query_heads||s.tokens<1||s.tokens>8192)
        throw std::invalid_argument("GDN register column geometry/operands");
    if(bytes<gdn_float_value_workspace(s))throw std::invalid_argument("GDN register column workspace");
    ++gdn_register_submissions();auto work=gdn_float_value_work(storage,s);
    gdn_register_prepare(x,s,work,stream);
    if(gdn_delta_active()){++gdn_delta_submissions();gdn_delta_half(x,s,work,out,state,stream);}
    else gdn_float_value_scan(x,s,work,out,state,stream);
}
}
