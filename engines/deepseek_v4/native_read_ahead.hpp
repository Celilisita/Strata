#pragma once
#include "router_prefetch.hpp"
#include "mxfp4_cpu_service.hpp"
#include "device_cache_protocol.hpp"
#include <array>
#include <atomic>
struct llama_model;
namespace strata::deepseek_v4 {
struct NativeReadAheadMetrics {
    uint64_t jobs=0,bytes=0,wall_ns=0,canceled=0,cancel_tail_ns=0;
    uint64_t predicted_ids=0,correct_ids=0,hash_jobs=0,hash_bytes=0;
    uint64_t completed_jobs=0,completed_first_expert_jobs=0,canceled_after_completion=0,observed_windows=0,unobserved_windows=0;
    uint64_t selected_one_jobs=0,selected_two_jobs=0;
    std::array<uint64_t,43> layer_window_ns{},layer_read_ns{},layer_window_count{};
};
class NativeReadAhead {
public:
    NativeReadAhead(ggml_backend_t,const llama_model&,Mxfp4CpuService&,DeviceMissPacket*,
        uint32_t(*)(const DeviceMissPacket*),const std::atomic<bool>&,const std::atomic<bool>&);
    bool poll(uint32_t real_ticket);
    void observe(const DeviceMissPacket&);
    NativeReadAheadMetrics metrics() const {return stats_;}
private:
    static bool guard(void*);
    bool read(const RouterPrefetchRoutes&,uint32_t real_ticket,bool hash);
    RouterPrefetchPacket* packet_=nullptr;
    const llama_model& model_;Mxfp4CpuService& cpu_;DeviceMissPacket* real_;
    uint32_t(*ready_)(const DeviceMissPacket*);const std::atomic<bool>& stop_;const std::atomic<bool>& active_;
    uint32_t seen_=0,hash_seen_=0,guard_ticket_=0,guard_sequence_=0;
    bool guard_hash_=false,select_two_=false,window_limited_=false;int hash_next_=3,last_hash_done_=-1;
    std::array<RouterPrefetchRoutes,3> hashes_{};
    std::array<uint32_t,43> epochs_{};
    std::array<std::array<bool,256>,43> touched_{};
    std::array<uint64_t,43> window_start_ns_{},window_read_ns_{};
    NativeReadAheadMetrics stats_;
};
}
