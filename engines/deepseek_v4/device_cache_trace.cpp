#include "device_cache_trace.hpp"
#include "device_cache_protocol.hpp"
#include "layer_snapshots_impl.hpp"
#include "attention_repeat_impl.hpp"
#include <chrono>
#include <thread>

namespace strata::deepseek_v4 {
namespace {
struct Entry { std::atomic<void*> context{nullptr}; std::atomic<CacheGraphTrace*> trace{nullptr}; };
std::array<Entry, 16> entries;
std::atomic<uint32_t> bound{0};
thread_local CacheGraphTrace* actor = nullptr;
CacheGraphTrace* find(void* context) {
    if (!bound.load(std::memory_order_relaxed)) return nullptr;
    for (auto& entry : entries)
        if (entry.context.load(std::memory_order_acquire) == context)
            return entry.trace.load(std::memory_order_acquire);
    return nullptr;
}
}
uint64_t cache_trace_nanoseconds() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
CacheGraphTrace* cache_graph_current_trace() { return actor; }
bool bind_cache_graph_trace(void* context, CacheGraphTrace* trace) {
    for (auto& entry : entries) {
        if (entry.context.load(std::memory_order_acquire) != context) continue;
        if (trace) return false; // One observer per canonical backend.
        entry.trace.store(nullptr, std::memory_order_release);
        entry.context.store(nullptr, std::memory_order_release);
        bound.fetch_sub(1, std::memory_order_relaxed);
        actor = nullptr;
        return true;
    }
    if (!trace) return false;
    for (auto& entry : entries) {
        void* empty = nullptr;
        if (!entry.context.compare_exchange_strong(empty, context)) continue;
        entry.trace.store(trace, std::memory_order_release);
        bound.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}
void cache_graph_phase(void* context, CacheGraphPhase phase, uint32_t stream) {
    actor = find(context);
    if (!actor) return;
    if(phase==CacheGraphPhase::Direct)++actor->graph_compute_calls;
    if(phase==CacheGraphPhase::Capture)++actor->graph_captures;
    if(phase==CacheGraphPhase::Launch)++actor->graph_launches;
    const uint64_t now=cache_trace_nanoseconds();
    if(actor->causal_phases&&phase==CacheGraphPhase::Enqueued&&
       actor->phase.load(std::memory_order_acquire)==uint32_t(CacheGraphPhase::Launch)) {
        const uint64_t before=actor->timestamp_ns.load(std::memory_order_relaxed);
        if(now<before)actor->graph_launch_host_errors.fetch_add(1,std::memory_order_relaxed);
        else {
            const uint64_t elapsed=now-before;actor->graph_launch_host_ns.fetch_add(elapsed,std::memory_order_relaxed);
            actor->graph_launch_host_count.fetch_add(1,std::memory_order_relaxed);
            auto maximum=actor->graph_launch_host_max_ns.load(std::memory_order_relaxed);
            while(elapsed>maximum&&!actor->graph_launch_host_max_ns.compare_exchange_weak(maximum,elapsed,std::memory_order_relaxed)){}
        }
    }
    actor->stream.store(stream, std::memory_order_relaxed);
    actor->timestamp_ns.store(now, std::memory_order_relaxed);
    actor->phase.store(uint32_t(phase), std::memory_order_release);
}
uint32_t cache_resource_begin(CacheGraphPhase phase) {
    if (!actor) return 0;
    return actor->phase.exchange(uint32_t(phase), std::memory_order_acq_rel);
}
void cache_graph_stream(uint32_t stream, uint32_t op, uint32_t node) {
    if (!actor) return;
    actor->stream.store(stream, std::memory_order_relaxed);
    actor->node_op.store(op, std::memory_order_relaxed);
    actor->node_index.store(node, std::memory_order_relaxed);
}
void cache_graph_miss_progress() {
    if (!actor || actor->phase.load(std::memory_order_relaxed) != uint32_t(CacheGraphPhase::Direct)) return;
    auto* packet = actor->miss_packet;
    if (!packet || packet->sequence == packet->response || packet->error) return;
    // CPU-only service5 performs no CUDA submission or bank mutation. Its
    // existing device join owns completion; eager enqueuing may continue.
    // GPU miss4 and every DMA-bearing provider retain the queue drain guard.
    if(actor->cpu_async_enqueue && packet->native_request==5)return;
    const auto start = cache_trace_nanoseconds();
    auto ticket_start = start;
    auto ticket = packet->sequence;
    actor->phase.store(uint32_t(CacheGraphPhase::MissProgress), std::memory_order_release);
    actor->enqueue_miss_waits.fetch_add(1, std::memory_order_relaxed);
    // A full eager launch queue can block DMA behind the waiting lookup.
    // Only a published miss parks this enqueuer, without calling CUDA.
    while (packet->sequence != packet->response && !packet->error) {
        const auto now = cache_trace_nanoseconds();
        if (packet->sequence != ticket) { ticket = packet->sequence; ticket_start = now; }
        if (now - ticket_start >= actor->miss_deadline_ns) {
            actor->enqueue_miss_deadlines.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        std::this_thread::yield();
    }
    actor->enqueue_miss_wait_ns.fetch_add(cache_trace_nanoseconds() - start, std::memory_order_relaxed);
    actor->phase.store(uint32_t(CacheGraphPhase::Direct), std::memory_order_release);
}
void cache_graph_enqueue_budget(uintptr_t stream) {
    if (actor && actor->phase.load(std::memory_order_relaxed) == uint32_t(CacheGraphPhase::Direct) &&
        actor->enqueue_budget) actor->enqueue_budget(actor->enqueue_user, stream);
}
void cache_resource_end(uint32_t prior, CacheGraphPhase resource, uint64_t start) {
    if (!actor) return;
    const size_t index = size_t(resource) - size_t(CacheGraphPhase::MemoryCreate);
    if (index >= cache_resource_count) return;
    const uint64_t elapsed = cache_trace_nanoseconds() - start;
    const uint64_t before = actor->resource_max_ns[index].load(std::memory_order_relaxed);
    if (elapsed > before) {
        actor->resource_begin_ns[index].store(start, std::memory_order_relaxed);
        actor->resource_end_ns[index].store(start + elapsed, std::memory_order_relaxed);
        actor->resource_stream[index].store(actor->stream.load(std::memory_order_relaxed), std::memory_order_relaxed);
        actor->resource_max_ns[index].store(elapsed, std::memory_order_release);
    }
    actor->phase.store(prior, std::memory_order_release);
}
CacheResourceGuard::CacheResourceGuard(CacheGraphPhase phase) : phase_(phase) {
    if (!actor) return;
    begin_ = cache_trace_nanoseconds();
    prior_ = cache_resource_begin(phase);
}
CacheResourceGuard::~CacheResourceGuard() {
    if (begin_) cache_resource_end(prior_, phase_, begin_);
}
}
