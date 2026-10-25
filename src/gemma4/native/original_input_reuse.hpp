#pragma once
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::gemma4::native {
inline bool original_input_reuse_requested(){
    static const bool enabled=[](){
        const char* value=std::getenv("STRATA_GEMMA_ORIGINAL_INPUT_REUSE");
        if(!value||std::strcmp(value,"0")==0)return false;
        if(std::strcmp(value,"1")==0)return true;
        throw std::invalid_argument("Gemma original input reuse requires0 or1");
    }();return enabled;
}
enum class OriginalInputCodec { NativeQ81, MmqDS4, MmqD4 };
inline OriginalInputCodec original_input_codec(int tokens,int type){
    if(tokens<1||(type!=12&&type!=14))throw std::invalid_argument("Gemma original input codec scope");
    if(tokens<=3)return OriginalInputCodec::NativeQ81;
    // The selected mmq_get_q8_1_ds_layout maps Q4_K to DS4, Q6_K to D4.
    return type==12?OriginalInputCodec::MmqDS4:OriginalInputCodec::MmqD4;
}
namespace original_input_reuse_detail {
struct Lease {
    const void* owner;const float* input;int tokens,columns;cudaStream_t stream;
    void* encoded=nullptr;OriginalInputCodec codec=OriginalInputCodec::NativeQ81;
    bool valid=true,ready=false;
    unsigned quantizations=0,hits=0,invalidations=0;
};
// This points only to a live stack scope; it never retains an input across groups.
inline thread_local Lease* current=nullptr;
inline bool matches(Lease& x,const float* input,int tokens,int columns,cudaStream_t stream,void* encoded){
    if(!x.valid)return false;
    if(x.input!=input||x.tokens!=tokens||x.columns!=columns||x.stream!=stream||(x.encoded&&x.encoded!=encoded)){
        x.valid=false;x.ready=false;++x.invalidations;return false;
    }return true;
}
}
// Caller promises immutable F32 input within this explicit producer/consumer
// group. Same pointer in a later group is always a new generation. No device
// pointer, allocation, stream, Graph or encoded-buffer ownership is transferred.
class OriginalInputReuseScope {
    original_input_reuse_detail::Lease lease_;
    bool enabled_;
public:
    OriginalInputReuseScope(const void* owner,const float* input,int tokens,int columns,cudaStream_t stream,bool enabled):
        lease_{owner,input,tokens,columns,stream},enabled_(enabled){
        if(auto* outer=original_input_reuse_detail::current){
            outer->valid=false;outer->ready=false;++outer->invalidations;
            throw std::logic_error("Gemma original input scopes cannot nest");
        }
        if(!enabled_)return;
        if(!owner||!input||tokens<1||columns<1||!stream)throw std::invalid_argument("Gemma original input scope lease");
        original_input_reuse_detail::current=&lease_;
    }
    ~OriginalInputReuseScope(){if(enabled_){lease_.ready=false;lease_.valid=false;original_input_reuse_detail::current=nullptr;}}
    OriginalInputReuseScope(const OriginalInputReuseScope&)=delete;
    OriginalInputReuseScope& operator=(const OriginalInputReuseScope&)=delete;
    unsigned quantizations()const{return lease_.quantizations;}
    unsigned hits()const{return lease_.hits;}
    unsigned invalidations()const{return lease_.invalidations;}
};
inline bool original_input_reuse_hit(const void* owner,const float* input,int tokens,int columns,cudaStream_t stream,void* encoded,OriginalInputCodec codec){
    auto* lease=original_input_reuse_detail::current;
    if(!lease||lease->owner!=owner)return false;
    if(!original_input_reuse_detail::matches(*lease,input,tokens,columns,stream,encoded))return false;
    if(lease->ready&&lease->codec==codec){++lease->hits;return true;}
    // A codec change clobbers the single encoded arena and requires a new producer.
    lease->ready=false;return false;
}
inline void original_input_reuse_publish(const void* owner,const float* input,int tokens,int columns,cudaStream_t stream,void* encoded,OriginalInputCodec codec){
    auto* lease=original_input_reuse_detail::current;
    if(!lease||lease->owner!=owner)return;
    ++lease->quantizations; // Actual producer submitted, even after scope invalidation.
    if(!original_input_reuse_detail::matches(*lease,input,tokens,columns,stream,encoded))return;
    lease->encoded=encoded;lease->codec=codec;lease->ready=true;
}
}
