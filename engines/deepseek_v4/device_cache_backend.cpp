#include "device_cache_backend.hpp"
#include "../glm5_next/partial_down_protocol.hpp"
#include "expert_frequency.hpp"
#include "../glm5_next/cuda_host_wait.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>
#ifdef STRATA_DEEPSEEK_DEVICE_CACHE_CUDA
#include "strata/core/dma_batch.hpp"
#include <cuda.h>
#endif

namespace strata::deepseek_v4 {
#ifdef STRATA_DEEPSEEK_DEVICE_CACHE_CUDA
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("device cache DMA: ") + cudaGetErrorString(status));
}
uint64_t elapsed(std::chrono::steady_clock::time_point begin) {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count());
}
}
struct DeviceCacheDma::Impl {
    int device = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t events[2]{};
    cudaEvent_t native_ready = nullptr;
    cudaEvent_t budget_events[4]{};
    size_t budget_head = 0, budget_size = 0, budget_nodes = 0;
    bool budget_active = true;
    void* slabs[2]{};
    bool busy[2]{};
    size_t used[2]{};
    bool pack_fragments = false;
    size_t next = 0;
    DeviceMissPacket* packet = nullptr;
    void* device_packet = nullptr;
    strata::glm5_next::PartialDownPhase* partial_phase = nullptr;
    void* device_partial_phase = nullptr;
    int32_t* directory = nullptr;
    uint32_t* frequency = nullptr;
    uint64_t timeout = 0;
    ExpertStaging* registration = nullptr;
    ExpertStagingMetrics metrics;
    DeviceCacheDmaCosts costs;
    CacheGraphTrace trace;
    ggml_backend_t backend = nullptr;
    bool (*trace_bind)(ggml_backend_t, CacheGraphTrace*) = nullptr;
    void completed_slabs() {
        busy[0] = busy[1] = false;
        if (pack_fragments) { used[0] = used[1] = 0; next = 0; }
    }
    template<class Call> cudaError_t call(size_t id, Call invoke) {
        CacheApiSpan span;
        span.phase_before = trace.phase.load(std::memory_order_acquire);
        span.owner_stream_before = trace.stream.load(std::memory_order_relaxed);
        span.owner_op_before = trace.node_op.load(std::memory_order_relaxed);
        span.owner_node_before = trace.node_index.load(std::memory_order_relaxed);
        span.begin_ns = cache_trace_nanoseconds();
        const auto result = invoke();
        span.end_ns = cache_trace_nanoseconds();
        span.phase_after = trace.phase.load(std::memory_order_acquire);
        span.owner_stream_after = trace.stream.load(std::memory_order_relaxed);
        span.owner_op_after = trace.node_op.load(std::memory_order_relaxed);
        span.owner_node_after = trace.node_index.load(std::memory_order_relaxed);
        span.thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
        span.copy_stream = reinterpret_cast<uintptr_t>(stream);
        span.max_ns = span.end_ns - span.begin_ns;
        if (span.max_ns > costs.api[id].max_ns) costs.api[id] = span;
        return result;
    }
    void budget(cudaStream_t compute_stream) {
        if (!budget_active || ++budget_nodes % 64) return;
        if (budget_size == 4) {
            const auto start = std::chrono::steady_clock::now();
            bool waited = false;
            for (;;) {
                // Keep the driver queue bounded before it can block an
                // enqueue while a newly published miss needs its resources.
                // Poll completion, never synchronize this waiting stream.
                cache_graph_miss_progress();
                ++costs.enqueue_budget_queries;
                const auto status = call(10, [&] { return cudaEventQuery(budget_events[budget_head]); });
                if (status == cudaSuccess) break;
                if (status != cudaErrorNotReady) check(status);
                if (!waited) { ++costs.enqueue_budget_waits; waited = true; }
                if (elapsed(start) >= timeout * 4) throw std::runtime_error("bounded CUDA enqueue completion timed out");
                std::this_thread::yield();
            }
            if (waited) costs.enqueue_budget_wait_ns += elapsed(start);
            budget_head = (budget_head + 1) % 4;
            --budget_size;
        }
        const size_t slot = (budget_head + budget_size) % 4;
        check(call(9, [&] { return cudaEventRecord(budget_events[slot], compute_stream); }));
        ++costs.enqueue_budget_records;
        ++budget_size;
    }
    ~Impl() {
        if (device >= 0) cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (backend && trace_bind) trace_bind(backend, nullptr);
        for (int i = 0; i < 2; ++i) {
            if (events[i]) cudaEventDestroy(events[i]);
            if (slabs[i]) cudaFreeHost(slabs[i]);
        }
        for (auto event : budget_events) if (event) cudaEventDestroy(event);
        if (native_ready) cudaEventDestroy(native_ready);
        if (directory) cudaFreeHost(directory);
        if (frequency) cudaFreeHost(frequency);
        if (partial_phase) cudaFreeHost(partial_phase);
        if (packet) cudaFreeHost(packet);
        if (stream) cudaStreamDestroy(stream);
    }
};
DeviceCacheDma::DeviceCacheDma(ggml_backend_t backend, ExpertStaging* registration,
                             bool prepare_all_streams, bool blocking_host_wait,
                             bool pack_fragments,bool causal_phases) : impl_(new Impl) {
    impl_->trace.causal_phases=causal_phases;
    const auto device = ggml_backend_get_device(backend);
    const auto reg = ggml_backend_dev_backend_reg(device);
    if (!reg || std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0)
        throw std::runtime_error("device cache requires the selected CUDA backend");
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i)
        if (ggml_backend_reg_dev_get(reg, i) == device) impl_->device = int(i);
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (impl_->device < 0 || impl_->device >= count) throw std::runtime_error("selected CUDA cache device differs");
    check(cudaSetDevice(impl_->device));
    CUmoduleLoadingMode mode;
    if (cuModuleGetLoadingMode(&mode) != CUDA_SUCCESS || mode != CU_MODULE_EAGER_LOADING)
        throw std::runtime_error("device cache requires EAGER modules before CUDA initialization; create a new runtime process");
    using Trace = bool (*)(ggml_backend_t, CacheGraphTrace*);
    impl_->trace_bind = reinterpret_cast<Trace>(ggml_backend_reg_get_proc_address(reg, "strata_device_cache_trace"));
    if (!impl_->trace_bind || !impl_->trace_bind(backend, &impl_->trace))
        throw std::runtime_error("cannot observe selected canonical CUDA backend");
    impl_->backend = backend;
    using Reserve = bool (*)(ggml_backend_t, size_t, bool);
    const auto reserve = reinterpret_cast<Reserve>(ggml_backend_reg_get_proc_address(reg, "strata_device_cache_reserve"));
    if (!reserve || !reserve(backend, 128ull << 20, prepare_all_streams))
        throw std::runtime_error("device cache requires a preallocated canonical CUDA VMM workspace");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, impl_->device));
    if (!props.canMapHostMemory)
        throw std::runtime_error("selected cache device lacks mapped doorbell support");
    // Cold mapped-file pages consumed 611ms in the real layer, versus <4ms
    // for resident RAM. Allow that measured source fault with a bounded margin.
    impl_->timeout = 1500000000; // Global timer nanoseconds; independent of DVFS.
    impl_->registration = registration;
    impl_->pack_fragments = pack_fragments;
    // The diagnostic may share all CCD logical processors with CPU workers.
    // Reuse these completion events with blocking waits; production defaults
    // and every event/stream dependency remain unchanged.
    const unsigned event_flags=cudaEventDisableTiming|(blocking_host_wait?cudaEventBlockingSync:0);
    int least=0,greatest=0;check(cudaDeviceGetStreamPriorityRange(&least,&greatest));
    check(cudaStreamCreateWithPriority(&impl_->stream,cudaStreamNonBlocking,least));
    for (int i = 0; i < 2; ++i) {
        check(cudaHostAlloc(&impl_->slabs[i], ExpertStaging::slab_bytes, cudaHostAllocPortable));
        check(cudaEventCreateWithFlags(&impl_->events[i], event_flags));
    }
    for (auto& event : impl_->budget_events) check(cudaEventCreateWithFlags(&event, event_flags));
    check(cudaEventCreateWithFlags(&impl_->native_ready, event_flags));
    impl_->trace.enqueue_user = impl_.get();
    impl_->trace.enqueue_budget = [](void* data, uintptr_t stream) {
        static_cast<Impl*>(data)->budget(reinterpret_cast<cudaStream_t>(stream));
    };
    check(cudaHostAlloc(reinterpret_cast<void**>(&impl_->packet), sizeof(DeviceMissPacket), cudaHostAllocMapped));
    new (impl_->packet) DeviceMissPacket{};
    impl_->trace.miss_packet = impl_->packet;
    impl_->trace.cpu_async_enqueue=strata::glm5_next::cpu_async_enqueue_requested();
    impl_->trace.miss_deadline_ns = impl_->timeout;
    check(cudaHostGetDevicePointer(&impl_->device_packet, impl_->packet, 0));
    check(cudaHostAlloc(reinterpret_cast<void**>(&impl_->directory), device_cache_max_layers * device_cache_max_experts * sizeof(int32_t), cudaHostAllocPortable));
    if (expert_frequency_enabled())
        check(cudaHostAlloc(reinterpret_cast<void**>(&impl_->frequency), (device_cache_max_layers*device_cache_max_experts+1) * sizeof(uint32_t), cudaHostAllocPortable));
}
DeviceCacheDma::~DeviceCacheDma() = default;
void DeviceCacheDma::prepare_partial_down_diagnostic() {
    if(impl_->partial_phase)return;
    check(cudaSetDevice(impl_->device));ggml_backend_synchronize(impl_->backend);
    strata::glm5_next::PartialDownPhase* phase=nullptr;void* alias=nullptr;
    check(cudaHostAlloc(reinterpret_cast<void**>(&phase),sizeof(*phase),cudaHostAllocMapped));
    try {check(cudaHostGetDevicePointer(&alias,phase,0));new (phase) strata::glm5_next::PartialDownPhase{};
        if(reinterpret_cast<uintptr_t>(phase)%64||reinterpret_cast<uintptr_t>(alias)%64)
            throw std::runtime_error("partial Down mapped phase alignment differs");
    }catch(...){cudaFreeHost(phase);throw;}
    impl_->partial_phase=phase;impl_->device_partial_phase=alias;
}
strata::glm5_next::PartialDownPhase* DeviceCacheDma::partial_down_phase() const noexcept {return impl_->partial_phase;}
uintptr_t DeviceCacheDma::partial_down_device_phase() const noexcept {return reinterpret_cast<uintptr_t>(impl_->device_partial_phase);}
DeviceMissPacket* DeviceCacheDma::packet() const noexcept { return impl_->packet; }
uintptr_t DeviceCacheDma::device_packet() const noexcept { return reinterpret_cast<uintptr_t>(impl_->device_packet); }
uint64_t DeviceCacheDma::timeout_ns() const noexcept { return impl_->timeout; }
ExpertStagingMetrics DeviceCacheDma::metrics() const noexcept { return impl_->metrics; }
DeviceCacheDmaCosts DeviceCacheDma::costs() const noexcept {
    auto result = impl_->costs;
    result.graph_compute_calls=impl_->trace.graph_compute_calls.load();
    result.graph_captures=impl_->trace.graph_captures.load();
    result.graph_launches=impl_->trace.graph_launches.load();
    result.causal_launch_host_enabled=impl_->trace.causal_phases;
    result.graph_launch_host_ns=impl_->trace.graph_launch_host_ns.load(std::memory_order_relaxed);
    result.graph_launch_host_max_ns=impl_->trace.graph_launch_host_max_ns.load(std::memory_order_relaxed);
    result.graph_launch_host_count=impl_->trace.graph_launch_host_count.load(std::memory_order_relaxed);
    result.graph_launch_host_errors=impl_->trace.graph_launch_host_errors.load(std::memory_order_relaxed);
    result.graph_preuploads = impl_->trace.graph_preuploads.load(std::memory_order_relaxed);
    result.graph_preupload_ns = impl_->trace.graph_preupload_ns.load(std::memory_order_relaxed);
    result.enqueue_miss_waits = impl_->trace.enqueue_miss_waits.load(std::memory_order_relaxed);
    result.enqueue_miss_wait_ns = impl_->trace.enqueue_miss_wait_ns.load(std::memory_order_relaxed);
    result.enqueue_miss_deadlines = impl_->trace.enqueue_miss_deadlines.load(std::memory_order_relaxed);
    for (size_t i = 0; i < cache_resource_count; ++i) {
        result.resource_max_ns[i] = impl_->trace.resource_max_ns[i].load(std::memory_order_acquire);
        result.resource_begin_ns[i] = impl_->trace.resource_begin_ns[i].load(std::memory_order_relaxed);
        result.resource_end_ns[i] = impl_->trace.resource_end_ns[i].load(std::memory_order_relaxed);
        result.resource_stream[i] = impl_->trace.resource_stream[i].load(std::memory_order_relaxed);
    }
    return result;
}

void DeviceCacheDma::bind_thread() {
    check(cudaSetDevice(impl_->device));
    check(cudaFree(nullptr));
}
void DeviceCacheDma::set_active(bool active) {
    impl_->trace.graph_misses.store(active, std::memory_order_release);
    if (active == impl_->budget_active) return;
    // Activating caller synchronizes compute first. Deactivation leaves
    // pending events alone; they are reused only at the next activation.
    if (active) impl_->budget_head = impl_->budget_size = impl_->budget_nodes = 0;
    impl_->budget_active = active;
}
void DeviceCacheDma::upload(ggml_tensor* dst, const void* source, size_t offset, size_t bytes) {
    if (!dst || !dst->buffer || !dst->data || !source || offset > ggml_nbytes(dst) || bytes > ggml_nbytes(dst) - offset)
        throw std::runtime_error("device cache upload range differs");
    auto* input = static_cast<const uint8_t*>(source);
    auto* output = static_cast<uint8_t*>(dst->data) + offset;
    while (bytes) {
        const auto* packed=impl_->registration?impl_->registration->packed_store():nullptr;
        const auto projection=packed?packed->project(input,bytes):PackedExpertStore::Projection{nullptr,0};
        if(packed&&!projection.data)throw std::runtime_error("cache upload outside packed expert inventory");
        auto segment = impl_->registration ? impl_->registration->transport_segment(input, bytes) :
                                            ExpertSourcePin::Segment{bytes, false};
        if(packed)segment={projection.bytes,true};
        size_t chunk = std::min(segment.bytes, segment.registered ? bytes : ExpertStaging::slab_bytes);
        if (!chunk) throw std::runtime_error("device cache source segment is empty");
        const void* dma_source = packed?projection.data:input;
        // Legacy advances on every segment. Packed direct uploads never touch
        // the fragment ring; only a filled slab advances its current slot.
        const size_t slot = impl_->pack_fragments ? impl_->next % 2 : impl_->next++ % 2;
        if (!segment.registered) {
            const bool reuse = !impl_->pack_fragments || impl_->used[slot] == ExpertStaging::slab_bytes;
            if (reuse && impl_->busy[slot]) {
                const auto begin = std::chrono::steady_clock::now();
                check(impl_->call(5, [&] { return cudaEventSynchronize(impl_->events[slot]); }));
                impl_->metrics.slab_wait_wall_ns += elapsed(begin);
                ++impl_->metrics.slab_waits;
            }
            if (reuse && impl_->pack_fragments) impl_->used[slot] = 0;
            const size_t slab_offset = impl_->pack_fragments ? impl_->used[slot] : 0;
            chunk = std::min(chunk, ExpertStaging::slab_bytes - slab_offset);
            auto* staging = static_cast<uint8_t*>(impl_->slabs[slot]) + slab_offset;
            const auto begin = std::chrono::steady_clock::now();
            std::memcpy(staging, input, chunk);
            impl_->metrics.host_copy_wall_ns += elapsed(begin);
            impl_->metrics.host_copy_bytes += chunk;
            dma_source = staging;
        }
        void* destinations[]{output};
        const void* sources[]{dma_source};
        const size_t sizes[]{chunk};
        // Native Strata transport, mode zero: no retry after a partial batch submission.
        const auto submit_begin = std::chrono::steady_clock::now();
        check(impl_->call(0, [&] { return strata::core::copy_blobs(destinations, sources, sizes, 1, impl_->stream, 0); }));
        if (!segment.registered) {
            // All fragments use this one stream. Its latest event covers
            // every earlier slice, while appends write only the unused tail.
            check(impl_->call(1, [&] { return cudaEventRecord(impl_->events[slot], impl_->stream); }));
            impl_->busy[slot] = true;
            if (impl_->pack_fragments) {
                impl_->used[slot] += chunk;
                if (impl_->used[slot] == ExpertStaging::slab_bytes) ++impl_->next;
            }
        } else {
            impl_->metrics.direct_bytes += chunk;
            ++impl_->metrics.direct_calls;
        }
        impl_->costs.submit_ns += elapsed(submit_begin);
        impl_->metrics.bytes += chunk;
        ++impl_->metrics.calls;
        input += chunk;
        output += chunk;
        bytes -= chunk;
    }
}
void DeviceCacheDma::commit(void* destination, const int32_t* directory, size_t count) {
    if(!directory || count > device_cache_max_layers*device_cache_max_experts)
        throw std::runtime_error("device directory projection exceeds its bounded allocation");
    std::memcpy(impl_->directory, directory, count*sizeof(int32_t));
    const auto submit_begin = std::chrono::steady_clock::now();
    check(impl_->call(2, [&] { return cudaMemcpyAsync(destination, impl_->directory, count*sizeof(int32_t), cudaMemcpyHostToDevice, impl_->stream); }));
    check(impl_->call(3, [&] { return cudaEventRecord(impl_->events[0], impl_->stream); }));
    impl_->costs.submit_ns += elapsed(submit_begin);
    const auto wait_begin = std::chrono::steady_clock::now();
    check(impl_->call(4, [&] { return cudaEventSynchronize(impl_->events[0]); }));
    impl_->costs.commit_wait_ns += elapsed(wait_begin);
    impl_->completed_slabs();
}
void DeviceCacheDma::seal_background() {check(cudaEventRecord(impl_->native_ready,impl_->stream));}
bool DeviceCacheDma::background_ready() {
    const auto status=cudaEventQuery(impl_->native_ready);
    if(status==cudaErrorNotReady)return false;
    check(status);impl_->completed_slabs();return true;
}
void DeviceCacheDma::wait_background() {
    check(cudaEventSynchronize(impl_->native_ready));
    impl_->completed_slabs();
}
void DeviceCacheDma::publish_native_slots(bool payload_pending) {
    if (!impl_->packet->gpu_slots || !impl_->packet->native_request)
        throw std::runtime_error("native TG slot destination absent");
    // The bank-free device publication kernel consumes reply_slots after the
    // response ticket. No DMA writes into a live MMVQ/publication allocation.
    // This event completes all real weight/directory writes before release.
    if (payload_pending) {
        const auto begin = std::chrono::steady_clock::now();
        check(impl_->call(11, [&] { return cudaEventRecord(impl_->native_ready, impl_->stream); }));
        impl_->costs.submit_ns += elapsed(begin);
        const auto wait = std::chrono::steady_clock::now();
        check(impl_->call(12, [&] { return cudaEventSynchronize(impl_->native_ready); }));
        impl_->costs.commit_wait_ns += elapsed(wait);
    }
    ++impl_->costs.native_slot_publish_calls;
    impl_->completed_slabs();
}
void DeviceCacheDma::read_frequency(const void* source, std::vector<uint32_t>& destination) {
    if (!source || !impl_->frequency) throw std::runtime_error("device frequency source or registration is absent");
    if (impl_->busy[0] || impl_->busy[1]) throw std::runtime_error("frequency snapshot precedes copy completion");
    const auto start = std::chrono::steady_clock::now();
    check(impl_->call(6, [&] { return cudaMemcpyAsync(impl_->frequency, source, destination.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost, impl_->stream); }));
    check(impl_->call(7, [&] { return cudaEventRecord(impl_->events[0], impl_->stream); }));
    check(impl_->call(8, [&] { return cudaEventSynchronize(impl_->events[0]); }));
    std::memcpy(destination.data(), impl_->frequency, destination.size()*sizeof(uint32_t));
    impl_->costs.frequency_read_ns += elapsed(start);
    impl_->costs.frequency_d2h_bytes += destination.size()*sizeof(uint32_t);
}
void DeviceCacheDma::read_miss_frequency(std::vector<uint32_t>& destination) {
    if (!impl_->frequency || impl_->packet->sequence == impl_->packet->response || impl_->packet->error)
        throw std::runtime_error("frequency snapshot lacks a published miss");
    const auto start = std::chrono::steady_clock::now();
    std::atomic_thread_fence(std::memory_order_acquire);
    std::memcpy(destination.data(), impl_->packet->frequency, destination.size()*sizeof(uint32_t));
    impl_->costs.frequency_read_ns += elapsed(start);
    impl_->costs.frequency_d2h_bytes += destination.size()*sizeof(uint32_t);
}
#else
struct DeviceCacheDma::Impl {};
DeviceCacheDma::DeviceCacheDma(ggml_backend_t, ExpertStaging*, bool, bool, bool, bool) { throw std::runtime_error("device cache needs CUDA"); }
DeviceCacheDma::~DeviceCacheDma() = default;
void DeviceCacheDma::prepare_partial_down_diagnostic(){throw std::runtime_error("partial Down requires CUDA");}
strata::glm5_next::PartialDownPhase* DeviceCacheDma::partial_down_phase() const noexcept{return nullptr;}
uintptr_t DeviceCacheDma::partial_down_device_phase() const noexcept{return 0;}
DeviceMissPacket* DeviceCacheDma::packet() const noexcept { return nullptr; }
uintptr_t DeviceCacheDma::device_packet() const noexcept { return 0; }
uint64_t DeviceCacheDma::timeout_ns() const noexcept { return 0; }
void DeviceCacheDma::bind_thread() {}
void DeviceCacheDma::set_active(bool) {}
void DeviceCacheDma::upload(ggml_tensor*, const void*, size_t, size_t) {}
void DeviceCacheDma::commit(void*, const int32_t*, size_t) {}
void DeviceCacheDma::seal_background() {}
bool DeviceCacheDma::background_ready() {return false;}
void DeviceCacheDma::wait_background() {}
void DeviceCacheDma::publish_native_slots(bool) {}
void DeviceCacheDma::read_frequency(const void*, std::vector<uint32_t>&) {}
void DeviceCacheDma::read_miss_frequency(std::vector<uint32_t>&) {}
ExpertStagingMetrics DeviceCacheDma::metrics() const noexcept { return {}; }
DeviceCacheDmaCosts DeviceCacheDma::costs() const noexcept { return {}; }
#endif
}
