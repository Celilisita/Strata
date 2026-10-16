#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace strata::deepseek_v4 {
struct DeviceMissPacket;
enum class CacheGraphPhase : uint32_t {
    Idle, Direct, Capture, Instantiate, Update, Launch, Enqueued, EndCapture,
    Prepare, MemoryCreate, MemoryMap, StreamSync, EventSync, TensorSet, MissProgress
};
inline constexpr std::size_t cache_resource_count = 5;
struct CacheGraphTrace {
    std::atomic<uint32_t> phase{0}, stream{0};
    std::atomic<uint32_t> node_op{0}, node_index{0};
    std::atomic<uint64_t> timestamp_ns{0};
    std::array<std::atomic<uint64_t>, cache_resource_count> resource_max_ns{};
    std::array<std::atomic<uint64_t>, cache_resource_count> resource_begin_ns{}, resource_end_ns{};
    std::array<std::atomic<uint32_t>, cache_resource_count> resource_stream{};
    DeviceMissPacket* miss_packet = nullptr;
    uint64_t miss_deadline_ns = 0;
    std::atomic<uint64_t> enqueue_miss_waits{0}, enqueue_miss_wait_ns{0}, enqueue_miss_deadlines{0};
    void* enqueue_user = nullptr;
    void (*enqueue_budget)(void*, uintptr_t stream) = nullptr;
    std::atomic<bool> graph_misses{true};
    std::atomic<uint64_t> graph_preuploads{0}, graph_preupload_ns{0};
};
struct CacheApiSpan {
    uint64_t max_ns = 0, begin_ns = 0, end_ns = 0, thread = 0;
    uintptr_t copy_stream = 0;
    uint32_t phase_before = 0, phase_after = 0, owner_stream_before = 0, owner_stream_after = 0;
    uint32_t owner_op_before = 0, owner_op_after = 0, owner_node_before = 0, owner_node_after = 0;
};
bool bind_cache_graph_trace(void* backend_context, CacheGraphTrace*);
void cache_graph_phase(void* backend_context, CacheGraphPhase, uint32_t stream);
void cache_graph_stream(uint32_t stream, uint32_t op = 0, uint32_t node = 0);
void cache_graph_miss_progress();
void cache_graph_enqueue_budget(uintptr_t stream);
CacheGraphTrace* cache_graph_current_trace();
uint32_t cache_resource_begin(CacheGraphPhase);
void cache_resource_end(uint32_t prior_phase, CacheGraphPhase, uint64_t begin_ns);
uint64_t cache_trace_nanoseconds();
class CacheResourceGuard {
public:
    explicit CacheResourceGuard(CacheGraphPhase);
    ~CacheResourceGuard();
private:
    CacheGraphPhase phase_;
    uint32_t prior_ = 0;
    uint64_t begin_ = 0;
};
}
