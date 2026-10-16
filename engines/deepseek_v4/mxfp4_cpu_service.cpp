#include "mxfp4_cpu_service.hpp"
#include "mxfp4_fast_dot.hpp"
#include "cpu_poll_wait.hpp"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::deepseek_v4 {
struct Mxfp4CpuService::Impl {
    int threads;
    const ggml_type_traits_cpu *dot, *quant;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake, finished;
    std::atomic<bool> stop{false};
    std::atomic<bool> busy{false};
    bool spin_wait=false;
    std::atomic<bool> spin_active{true};
    std::atomic<uint64_t> generation{0};
    std::atomic<int> arrived{0}, phase{0};
    Mxfp4CpuBank bank;
    std::array<int32_t, 12> ids{};
    bool paired=false,fast=false;
    bool read_only=false;
    bool read_serial_top2=false;
    uint64_t read_deadline_ns=0;
    IdleGuard read_guard=nullptr;void* read_user=nullptr;
    std::array<int32_t,2> read_selected{{-1,-1}};
    std::array<uint64_t,32> read_bytes{},read_checksums{},read_first_bytes{};
    std::array<uint8_t,32> read_finished_valid{};
    int dual_cost=200;
    std::array<std::array<int,12>,2> ranks{};
    alignas(64) std::array<uint8_t,2*4352> pair_xq{};
    alignas(64) std::array<uint8_t,24*2176> pair_hq{};
    alignas(64) std::array<float,24*2048> pair_h{};
    int count = 0;
    // D4 Q8_0 activations need 4352/2176 bytes, larger than the old Qwen
    // native expert pool's 4096/1024 byte bounds. These are explicit D4 buffers.
    alignas(64) std::array<uint8_t, 4352> xq{};
    alignas(64) std::array<uint8_t, 6 * 2176> hq{};
    alignas(64) std::array<float, 6 * 2048> h{};
    float* output = nullptr;
    Completion completion = nullptr;
    void* user = nullptr;
    explicit Impl(int n) : threads(n) {
        if (n < 1 || n > 32) throw std::runtime_error("D4 CPU service thread budget invalid");
        const char* spin=std::getenv("STRATA_DSV4_CPU_SPIN_WAIT");
        if(spin&&std::strcmp(spin,"0")&&std::strcmp(spin,"1"))throw std::runtime_error("CPU spin wait must be0 or1");
        spin_wait=spin&&!std::strcmp(spin,"1");
        const char* serial=std::getenv("STRATA_DSV4_PREFETCH_SERIAL_TOP2");
        read_serial_top2=serial&&!std::strcmp(serial,"1");
        ggml_cpu_init();const char* fast_setting=std::getenv("STRATA_DSV4_CPU_VNNI");fast=fast_setting&&std::strcmp(fast_setting,"1")==0&&mxfp4_fast_dot_supported();
        if(const char* value=std::getenv("STRATA_DSV4_CPU_DUAL_COST_PCT")){char* end=nullptr;const auto cost=std::strtol(value,&end,10);
            if(!*value||*end||cost<100||cost>200)throw std::runtime_error("CPU dual-position row cost must be100..200 percent");
            if(fast)dual_cost=int(cost);}
        dot = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
        if (!dot || !dot->vec_dot || dot->vec_dot_type != GGML_TYPE_Q8_0 || dot->nrows != 1)
            throw std::runtime_error("D4 CPU service MXFP4 row-dot contract differs");
        quant = ggml_get_type_traits_cpu(dot->vec_dot_type);
        if (!quant || !quant->from_float || ggml_row_size(dot->vec_dot_type, 4096) != xq.size() ||
            ggml_row_size(dot->vec_dot_type, 2048) != hq.size() / 6)
            throw std::runtime_error("D4 CPU service activation quantizer geometry differs");
        try { for (int i = 0; i < n; ++i) workers.emplace_back(&Impl::worker, this, i); }
        catch (...) { shutdown(); throw; }
    }
    ~Impl() { shutdown(); }
    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            finished.wait(lock, [&] { return !busy; });
            stop.store(true,std::memory_order_release);
        }
        wake.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
        workers.clear();
    }
    void barrier() {
        const int before = phase.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == threads) {
            arrived.store(0, std::memory_order_relaxed);
            phase.fetch_add(1, std::memory_order_release);
        } else {
            while (phase.load(std::memory_order_acquire) == before) {
                if(spin_wait)cpu_poll_pause();else std::this_thread::yield();
            }
        }
    }
    void row_dot(int n,float* output,const void* weight,const void* activation) {if(fast)mxfp4_fast_dot(n,weight,activation,nullptr,output);else dot->vec_dot(n,output,0,weight,0,activation,0,1);}
    // Common experts perform two dots per row. Partition their weighted row
    // intervals rather than equal union rows; every weight row has one owner.
    template<class F> void pair_partition(int lane,int rows,F function) {
        const auto row_cost=[&](int e){return ranks[0][e]>=0&&ranks[1][e]>=0?dual_cost:100;};
        int work=0;for(int e=0;e<count;++e)work+=row_cost(e);
        const int begin=work*rows*lane/threads,end=work*rows*(lane+1)/threads;int prefix=0;
        for(int e=0;e<count;++e){const int cost=row_cost(e);
            const int first=std::max(0,std::min(rows,(begin-prefix+cost-1)/cost));
            const int last=std::max(0,std::min(rows,(end-prefix+cost-1)/cost));
            for(int row=first;row<last;++row)function(e,row);prefix+=rows*cost;}
    }
    void pair_rows(int lane) {
        pair_partition(lane,2048,[&](int e,int row) {
            const int id=ids[e];
            const auto* g=static_cast<const char*>(bank.gate->data)+id*bank.gate->nb[2]+row*bank.gate->nb[1];
            const auto* u=static_cast<const char*>(bank.up->data)+id*bank.up->nb[2]+row*bank.up->nb[1];
            std::array<float,2> gate{},up{};
            // Adjacent dot calls consume the same immutable 2176-byte row;
            // only activations differ. Reuse is within the worker's L1 cache.
            if(fast&&ranks[0][e]>=0&&ranks[1][e]>=0){mxfp4_fast_dot(4096,g,pair_xq.data(),pair_xq.data()+4352,gate.data());mxfp4_fast_dot(4096,u,pair_xq.data(),pair_xq.data()+4352,up.data());}
            else for(int t=0;t<2;++t)if(ranks[t][e]>=0){row_dot(4096,&gate[t],g,pair_xq.data()+t*4352);row_dot(4096,&up[t],u,pair_xq.data()+t*4352);}
            for(int t=0;t<2;++t)if(ranks[t][e]>=0) {
                if(bank.clamp>1e-6f){gate[t]=std::min(gate[t],bank.clamp);up[t]=std::max(std::min(up[t],bank.clamp),-bank.clamp);}
                pair_h[(t*12+e)*2048+row]=gate[t]/(1.0f+std::exp(-gate[t]))*up[t];
            }
        });
        barrier();
        for(int e=lane;e<count;e+=threads)for(int t=0;t<2;++t)if(ranks[t][e]>=0)
            quant->from_float(pair_h.data()+(t*12+e)*2048,pair_hq.data()+(t*12+e)*2176,2048);
        barrier();
        pair_partition(lane,4096,[&](int e,int row) {
            const auto* d=static_cast<const char*>(bank.down->data)+ids[e]*bank.down->nb[2]+row*bank.down->nb[1];
            if(fast&&ranks[0][e]>=0&&ranks[1][e]>=0){std::array<float,2> value{};mxfp4_fast_dot(2048,d,pair_hq.data()+e*2176,pair_hq.data()+(12+e)*2176,value.data());for(int t=0;t<2;++t)output[(t*6+ranks[t][e])*4096+row]=value[t];}
            else for(int t=0;t<2;++t)if(ranks[t][e]>=0)row_dot(2048,output+(t*6+ranks[t][e])*4096+row,d,pair_hq.data()+(t*12+e)*2176);
        });
    }
    void read_rows(int lane) {
        uint64_t checksum=0,bytes=0;bool expired=false;
        std::array<int,12> selected{};int selected_count=0;
        for(int e=0;e<count;++e)if(read_selected[0]<0||ids[e]==read_selected[0]||ids[e]==read_selected[1])selected[selected_count++]=e;
        const auto touch=[&](const ggml_tensor* tensor,int expert,int row) {
            if(expired)return;
            if(read_selected[0]>=0&&ids[expert]!=read_selected[0]&&ids[expert]!=read_selected[1])return;
            if(read_deadline_ns&&(row%32)==0) {
                const auto now=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                if(now>=read_deadline_ns||(read_guard&&!read_guard(read_user))) {expired=true;return;}
            }
            const auto* weight=static_cast<const char*>(tensor->data)+ids[expert]*tensor->nb[2]+row*tensor->nb[1];
            for(size_t offset=0;offset<tensor->nb[1];offset+=64)
                checksum^=*reinterpret_cast<const volatile uint64_t*>(weight+offset);
            bytes+=tensor->nb[1];
        };
        if(read_serial_top2&&selected_count==2){
            for(int i=0;i<2;++i){const int e=selected[i];
                for(int row=2048*lane/threads;row<2048*(lane+1)/threads;++row){touch(bank.gate,e,row);touch(bank.up,e,row);}
                for(int row=4096*lane/threads;row<4096*(lane+1)/threads;++row)touch(bank.down,e,row);
                barrier();if(i==0)read_first_bytes[lane]=bytes;
            }
        }else{
            pair_partition(lane,2048,[&](int e,int row) {touch(bank.gate,e,row);touch(bank.up,e,row);});
            pair_partition(lane,4096,[&](int e,int row) {touch(bank.down,e,row);});
            read_first_bytes[lane]=selected_count==1?bytes:0;
        }
        read_bytes[lane]=bytes;read_checksums[lane]=checksum;
        read_finished_valid[lane]=uint8_t(!expired&&(!read_guard||read_guard(read_user)));
    }
    void worker(int lane) {
        uint64_t seen = 0;
        for (;;) {
            if(spin_wait&&spin_active.load(std::memory_order_acquire)){
                while(generation.load(std::memory_order_acquire)==seen&&!stop.load(std::memory_order_acquire)&&
                      spin_active.load(std::memory_order_acquire))cpu_poll_pause();
                if(stop.load(std::memory_order_acquire))return;
                if(generation.load(std::memory_order_acquire)==seen)continue;
                seen=generation.load(std::memory_order_acquire);
            }else{
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stop.load(std::memory_order_acquire) || generation.load(std::memory_order_acquire) != seen ||
                    (spin_wait&&spin_active.load(std::memory_order_acquire)); });
                if (stop.load(std::memory_order_acquire)) return;
                if(generation.load(std::memory_order_acquire)==seen)continue;
                seen = generation.load(std::memory_order_acquire);
            }
            if(read_only)read_rows(lane);
            else if(paired)pair_rows(lane);
            else {
            const size_t gu_row = bank.gate->nb[1], dn_row = bank.down->nb[1];
            // Give each worker contiguous row ranges; no row/expert job heap
            // allocations and no activation quantization repeated per expert.
            const int gu_begin = count * 2048 * lane / threads;
            const int gu_end = count * 2048 * (lane + 1) / threads;
            for (int i = gu_begin; i < gu_end; ++i) {
                const int e = i / 2048, row = i % 2048, id = ids[e];
                const auto* g = static_cast<const char*>(bank.gate->data) + id * bank.gate->nb[2] + row * gu_row;
                const auto* u = static_cast<const char*>(bank.up->data) + id * bank.up->nb[2] + row * gu_row;
                float gate, up;
                row_dot(4096,&gate,g,xq.data());
                row_dot(4096,&up,u,xq.data());
                if (bank.clamp > 1e-6f) {
                    gate = std::min(gate, bank.clamp);
                    up = std::max(std::min(up, bank.clamp), -bank.clamp);
                }
                h[i] = gate / (1.0f + std::exp(-gate)) * up;
            }
            barrier();
            for (int e = lane; e < count; e += threads)
                quant->from_float(h.data() + e * 2048, hq.data() + e * 2176, 2048);
            barrier();
            const int dn_begin = count * 4096 * lane / threads;
            const int dn_end = count * 4096 * (lane + 1) / threads;
            for (int i = dn_begin; i < dn_end; ++i) {
                const int e = i / 4096, row = i % 4096;
                const auto* d = static_cast<const char*>(bank.down->data) + ids[e] * bank.down->nb[2] + row * dn_row;
                row_dot(2048,output+i,d,hq.data()+e*2176);
            }
            }
            barrier();
            if (lane == 0) {
                bool error = false;
                if(!read_only)for (int i = 0; i < (paired?12:count) * 4096; ++i) error |= !std::isfinite(output[i]);
                const auto done = completion;
                void* const done_user = user;
                { std::lock_guard<std::mutex> lock(mutex); busy = false; }
                finished.notify_one();
                if (done) done(done_user, error);
            }
        }
    }
};
uint64_t Mxfp4CpuService::worker_cpu_100ns() const {
#ifdef _WIN32
    uint64_t total=0;
    for(auto& worker:impl_->workers){
        FILETIME created{},exited{},kernel{},user{};
        if(GetThreadTimes(worker.native_handle(),&created,&exited,&kernel,&user)){
            const auto count=[](const FILETIME& value){return (uint64_t(value.dwHighDateTime)<<32)|value.dwLowDateTime;};
            total+=count(kernel)+count(user);
        }
    }
    return total;
#else
    return 0;
#endif
}
Mxfp4CpuService::Mxfp4CpuService(int threads) : impl_(std::make_unique<Impl>(threads)) {}
Mxfp4CpuService::~Mxfp4CpuService() = default;
void Mxfp4CpuService::submit(const Mxfp4CpuBank& bank, const float* x, const int32_t* ids,
                            size_t count, float* down, Completion done, void* user) {
    if (!x || !ids || !down || count < 1 || count > 6)
        throw std::runtime_error("D4 CPU service submission outside six original ranks");
    for (auto* t : {bank.gate, bank.up, bank.down})
        if (!t || !t->data || t->type != GGML_TYPE_MXFP4 || !ggml_is_contiguous_to_2(t) ||
            t->nb[2]<t->nb[1]*size_t(t->ne[1]) || t->ne[2] != 256 || t->ne[3] != 1)
            throw std::runtime_error("D4 CPU service requires contiguous MXFP4 expert matrices and valid expert strides");
    if (bank.gate->ne[0] != 4096 || bank.gate->ne[1] != 2048 || bank.up->ne[0] != 4096 || bank.up->ne[1] != 2048 ||
        bank.down->ne[0] != 2048 || bank.down->ne[1] != 4096)
        throw std::runtime_error("D4 CPU service bank dimensions differ");
    for (size_t i = 0; i < count; ++i)
        if (ids[i] < 0 || ids[i] >= 256 || std::find(ids, ids + i, ids[i]) != ids + i)
            throw std::runtime_error("D4 CPU service selected IDs invalid or duplicated");
    for (int i = 0; i < 4096; ++i)
        if (!std::isfinite(x[i])) throw std::runtime_error("D4 CPU service nonfinite activation");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->busy || impl_->stop) throw std::runtime_error("D4 CPU service overlapping submission");
    impl_->read_only=false;impl_->paired=false;impl_->bank = bank; impl_->count = int(count); impl_->output = down;
    impl_->completion = done; impl_->user = user;
    std::copy_n(ids, count, impl_->ids.begin());
    impl_->quant->from_float(x, impl_->xq.data(), 4096);
    impl_->busy = true; impl_->generation.fetch_add(1,std::memory_order_release);
    if(!impl_->spin_wait)impl_->wake.notify_all();
}
void Mxfp4CpuService::wait() {
    if(impl_->spin_wait){while(impl_->busy.load(std::memory_order_acquire))cpu_poll_pause();return;}
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->finished.wait(lock, [&] { return !impl_->busy; });
}
void Mxfp4CpuService::compute(const Mxfp4CpuBank& b, const float* x, const int32_t* ids, size_t count, float* out) {
    submit(b, x, ids, count, out); wait();
    for (size_t i = 0; i < count * 4096; ++i)
        if (!std::isfinite(out[i])) throw std::runtime_error("D4 CPU service nonfinite contribution");
}
size_t Mxfp4CpuService::compute_pair(const Mxfp4CpuBank& bank,const float* x,const int32_t* route,
                                  const uint32_t* masks,float* out) {
    if(!x||!route||!masks||!out||(masks[0]|masks[1])>63||!(masks[0]|masks[1]))throw std::runtime_error("MTP pair inputs invalid");
    for(auto* t:{bank.gate,bank.up,bank.down})if(!t||!t->data||t->type!=GGML_TYPE_MXFP4||t->ne[2]!=256||!ggml_is_contiguous_to_2(t)||t->nb[2]<t->nb[1]*size_t(t->ne[1]))throw std::runtime_error("MTP pair bank layout invalid");
    if(bank.gate->ne[0]!=4096||bank.gate->ne[1]!=2048||bank.up->ne[0]!=4096||bank.up->ne[1]!=2048||bank.down->ne[0]!=2048||bank.down->ne[1]!=4096)throw std::runtime_error("MTP pair expert dimensions invalid");
    std::array<int32_t,12> ids{};std::array<std::array<int,12>,2> ranks;for(auto& r:ranks)r.fill(-1);int count=0;
    for(int t=0;t<2;++t)for(int rank=0;rank<6;++rank) {
        const int id=route[t*6+rank];if(id<0||id>=256||std::find(route+t*6,route+t*6+rank,id)!=route+t*6+rank)throw std::runtime_error("MTP pair expert IDs invalid");
        if(!(masks[t]&(1u<<rank)))continue;
        auto found=std::find(ids.begin(),ids.begin()+count,id);int e=int(found-ids.begin());if(e==count)ids[count++]=id;ranks[t][e]=rank;
    }
    for(int i=0;i<8192;++i)if(!std::isfinite(x[i]))throw std::runtime_error("MTP pair activation invalid");
    std::fill_n(out,12*4096,0.f);
    {std::lock_guard<std::mutex> lock(impl_->mutex);if(impl_->busy||impl_->stop)throw std::runtime_error("MTP pair overlaps CPU service");
        impl_->read_only=false;impl_->paired=true;impl_->bank=bank;impl_->count=count;impl_->ids=ids;impl_->ranks=ranks;impl_->output=out;impl_->completion=nullptr;impl_->user=nullptr;
        for(int t=0;t<2;++t)impl_->quant->from_float(x+t*4096,impl_->pair_xq.data()+t*4352,4096);
        impl_->busy=true;impl_->generation.fetch_add(1,std::memory_order_release);if(!impl_->spin_wait)impl_->wake.notify_all();}
    wait();return size_t(count);
}
uint64_t Mxfp4CpuService::read_ahead_pair(const Mxfp4CpuBank& bank,const int32_t* route,const uint32_t* masks,double window_ms,const int32_t* selected,IdleGuard guard,void* user,bool* finished_while_valid,uint64_t* first_expert_bytes) {
    if(!route||!masks||(masks[0]|masks[1])>63||!(masks[0]|masks[1])||!std::isfinite(window_ms)||window_ms<0||window_ms>10)
        throw std::runtime_error("Read-ahead pair inputs invalid");
    for(auto* t:{bank.gate,bank.up,bank.down})if(!t||!t->data||t->type!=GGML_TYPE_MXFP4||t->ne[2]!=256||t->ne[3]!=1||!ggml_is_contiguous_to_2(t)||t->nb[2]<t->nb[1]*size_t(t->ne[1]))throw std::runtime_error("Read-ahead bank layout invalid");
    if(bank.gate->ne[0]!=4096||bank.gate->ne[1]!=2048||bank.up->ne[0]!=4096||bank.up->ne[1]!=2048||bank.down->ne[0]!=2048||bank.down->ne[1]!=4096)throw std::runtime_error("Read-ahead dimensions invalid");
    std::array<int32_t,12> ids{};std::array<std::array<int,12>,2> ranks;for(auto& r:ranks)r.fill(-1);int count=0;
    for(int t=0;t<2;++t)for(int rank=0;rank<6;++rank) {
        if(!(masks[t]&(1u<<rank)))continue;
        const int id=route[t*6+rank];if(id<0||id>=256)throw std::runtime_error("Read-ahead expert ID invalid");
        const int e=int(std::find(ids.begin(),ids.begin()+count,id)-ids.begin());if(e==count)ids[count++]=id;ranks[t][e]=rank;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);if(impl_->busy||impl_->stop)throw std::runtime_error("Read-ahead requires idle CPU pool");
        impl_->read_guard=guard;impl_->read_user=user;impl_->read_selected=selected?std::array<int32_t,2>{selected[0],selected[1]}:std::array<int32_t,2>{-1,-1};
        impl_->read_only=true;impl_->paired=true;impl_->bank=bank;impl_->count=count;impl_->ids=ids;impl_->ranks=ranks;impl_->output=nullptr;impl_->completion=nullptr;impl_->user=nullptr;
        const auto now=std::chrono::steady_clock::now().time_since_epoch();
        impl_->read_deadline_ns=window_ms?uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count())+uint64_t(window_ms*1e6):0;
        impl_->busy=true;impl_->generation.fetch_add(1,std::memory_order_release);if(!impl_->spin_wait)impl_->wake.notify_all();
    }
    wait();uint64_t bytes=0;bool valid=true;
    uint64_t first=0;for(int lane=0;lane<impl_->threads;++lane){bytes+=impl_->read_bytes[lane];first+=impl_->read_first_bytes[lane];valid&=impl_->read_finished_valid[lane]!=0;}
    if(finished_while_valid)*finished_while_valid=valid;
    if(first_expert_bytes)*first_expert_bytes=first;
    return bytes;
}
bool Mxfp4CpuService::idle() const {return !impl_->busy.load(std::memory_order_acquire);}
void Mxfp4CpuService::set_spin_active(bool active) {
    if(!impl_->spin_wait)return;
    if(impl_->busy.load(std::memory_order_acquire))throw std::runtime_error("CPU worker phase switch during active expert job");
    impl_->spin_active.store(active,std::memory_order_release);
    impl_->wake.notify_all();
}

}
