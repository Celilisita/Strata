#pragma once
#include "state_snapshot_copy.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <array>
#include <memory>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <atomic>
namespace strata::qwen35::native {
inline bool state_owner_requested(){static const bool enabled=[] {
    const char *v=std::getenv("STRATA_DENSE_MTP_STATE_OWNER");if(!v||!std::strcmp(v,"0"))return false;
    if(!std::strcmp(v,"1"))return true;throw std::invalid_argument("STRATA_DENSE_MTP_STATE_OWNER must be0or1");
}();return enabled;}
struct StateOwnerStats {uint64_t admissions=0,targets=0,commits=0,rollbacks=0,ingress_bytes=0,normalization_bytes=0;};
inline StateOwnerStats &state_owner_stats(){static thread_local StateOwnerStats stats;return stats;}
struct StateOwnerRow {int index;float **live;unsigned *history;float *a,*b;};
struct StateOwner {
    const void *model=nullptr,*mtp=nullptr;std::array<StateOwnerRow,48> rows{};unsigned anchor=0,previous=0;
    cudaStream_t stream=nullptr;size_t *counter=nullptr;bool active=false,pending=false;
    float *bank(const StateOwnerRow &r,unsigned p)const{return p==anchor?r.a:r.b;}
    void bind(unsigned p){if(p>1)throw std::logic_error("stateowner bank index");for(auto &r:rows)*r.live=bank(r,p);}
    void rollback(){if(!pending)throw std::logic_error("stateowner rollback outside target");bind(previous);
        for(auto &r:rows)*r.history=previous;pending=false;active=true;++state_owner_stats().rollbacks;}
    size_t move(bool canonical,unsigned parity){
        if(pending)rollback();bool needs=false,same=true;StateSnapshotCopyBatch batch{};
        for(size_t i=0;i<rows.size();++i){auto &r=rows[i];auto *dest=canonical?r.a:bank(r,parity);const bool different=*r.live!=dest;
            if(i) same&=different==needs;else needs=different;batch.rows[i]={dest,*r.live};}
        if(!same)throw std::logic_error("stateowner mixed layer owners");
        if(needs){cudaStreamCaptureStatus capture{};check_cuda(cudaStreamIsCapturing(stream,&capture),"stateowner transition capture status");
            if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("stateowner data transition must precede capture");
            state_snapshot_copy(batch,stream);check_cuda(cudaStreamSynchronize(stream),"stateowner real transition completion");
            const size_t bytes=48*StateSnapshotCopyBytes;if(counter)*counter+=bytes;
            if(canonical)state_owner_stats().normalization_bytes+=bytes;else state_owner_stats().ingress_bytes+=bytes;
        }
        for(auto &r:rows)*r.live=canonical?r.a:bank(r,parity);active=!canonical;
        return needs?48*StateSnapshotCopyBytes:0;
    }
    void normalize(){move(true,0);active=false;}
    void begin(unsigned p){if(pending)throw std::logic_error("stateowner repeated target");move(false,p);previous=p;pending=true;++state_owner_stats().targets;}
    void commit(){if(pending){pending=false;active=true;++state_owner_stats().commits;}}
};
struct StateOwnerThreads {std::mutex mutex;std::unordered_map<const void*,uint64_t> owners;};
inline StateOwnerThreads &state_owner_threads(){static auto *registry=new StateOwnerThreads;return *registry;}
inline uint64_t state_owner_thread_token(){static auto *next=new std::atomic<uint64_t>(1);
    static thread_local const uint64_t token=next->fetch_add(1);return token;}
inline void check_state_owner_thread(const void *model,bool enroll=false){auto &r=state_owner_threads();std::lock_guard<std::mutex> lock(r.mutex);
    auto it=r.owners.find(model);if(it!=r.owners.end()&&it->second!=state_owner_thread_token())
        throw std::logic_error("stateowner model accessed from a different execution thread");
    if(enroll&&it==r.owners.end())r.owners.emplace(model,state_owner_thread_token());}
inline void forget_state_owner_thread(const void *model){check_state_owner_thread(model);auto &r=state_owner_threads();std::lock_guard<std::mutex> lock(r.mutex);r.owners.erase(model);}
inline auto &state_owner_records(){static thread_local std::unordered_map<const void*,std::unique_ptr<StateOwner>> records;return records;}
inline StateOwner *state_owner(const void *model){check_state_owner_thread(model);auto &m=state_owner_records();auto i=m.find(model);return i==m.end()?nullptr:i->second.get();}
inline void normalize_state_owner(const void *model){if(auto *r=state_owner(model))r->normalize();}
inline void release_state_owner(const void *model){check_state_owner_thread(model);auto &m=state_owner_records();auto i=m.find(model);if(i!=m.end()){i->second->normalize();m.erase(i);forget_state_owner_thread(model);}}
inline void prepare_state_owner(const void *model,const void *mtp,const std::array<StateOwnerRow,48> &rows,unsigned anchor,cudaStream_t stream,size_t *counter){
    if(anchor>1)throw std::logic_error("stateowner actual ingress parity");
    check_state_owner_thread(model,true);
    auto &slot=state_owner_records()[model];if(slot){if(slot->pending)throw std::logic_error("stateowner capture during open transaction");slot->normalize();if(slot->mtp!=mtp)throw std::logic_error("stateowner borrowed bank owner changed");}
    else{slot=std::make_unique<StateOwner>();slot->model=model;slot->mtp=mtp;slot->rows=rows;}
    auto &r=*slot;r.anchor=anchor;r.stream=stream;r.counter=counter;r.active=r.pending=false;++state_owner_stats().admissions;
    constexpr size_t bytes=StateSnapshotCopyBytes;
    for(size_t i=0;i<48;++i){const auto &x=r.rows[i];
        if(!x.live||!x.history||!x.a||!x.b||x.index<0||x.index>=64||*x.live!=x.a)
            throw std::logic_error("stateowner cold borrowed binding");
        for(auto p:{x.a,x.b}){const auto begin=reinterpret_cast<uintptr_t>(p);if(begin%16||begin>std::numeric_limits<uintptr_t>::max()-bytes)throw std::logic_error("stateowner cold span");
            for(size_t j=0;j<=i;++j)for(auto q:{r.rows[j].a,r.rows[j].b})if(p!=q){const auto other=reinterpret_cast<uintptr_t>(q);
                if(begin<other+bytes&&other<begin+bytes)throw std::logic_error("stateowner overlapping banks");}}
        if(x.a==x.b)throw std::logic_error("stateowner aliased banks");
        for(size_t j=0;j<i;++j)if(x.index==r.rows[j].index||x.a==r.rows[j].a||x.a==r.rows[j].b||x.b==r.rows[j].a||x.b==r.rows[j].b)
            throw std::logic_error("stateowner duplicate layer binding");
    }
}
struct StateOwnerCapture;
inline StateOwnerCapture *&state_owner_capture(){static thread_local StateOwnerCapture *scope=nullptr;return scope;}
// mode0: original canonical A; mode1: target Bank[p]→Bank[p^1]; mode2: prefix-restore destination Bank[p].
struct StateOwnerCapture {
    StateOwner *record;unsigned parity;int mode;std::array<float*,48> previous{};
    StateOwnerCapture(const void *model,unsigned p,int kind):record(state_owner(model)),parity(p),mode(kind){
        if(state_owner_capture())throw std::logic_error("nested stateowner capture");if(!record)return;
        for(size_t i=0;i<48;++i){auto &r=record->rows[i];previous[i]=*r.live;*r.live=kind?record->bank(r,p):r.a;}
        state_owner_capture()=this;
    }
    StateOwnerCapture(const StateOwnerCapture&)=delete;StateOwnerCapture& operator=(const StateOwnerCapture&)=delete;
    ~StateOwnerCapture(){if(record){for(size_t i=0;i<48;++i)*record->rows[i].live=previous[i];state_owner_capture()=nullptr;}}
};
inline bool state_owner_target_capture(const void *model){check_state_owner_thread(model);auto *s=state_owner_capture();return s&&s->record&&s->record->model==model&&s->mode==1;}
inline float *state_owner_output(const void *model,int index,float *input){auto *s=state_owner_capture();
    if(!state_owner_target_capture(model))return input;
    for(const auto &r:s->record->rows)if(r.index==index){if(input!=s->record->bank(r,s->parity))throw std::logic_error("stateowner target input bank drift");return s->record->bank(r,s->parity^1);}
    throw std::logic_error("stateowner unknown recurrent layer");
}
}
