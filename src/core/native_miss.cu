#include "strata/core/native_miss.hpp"
#include "strata/core/native_miss_pickup.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>

namespace strata::core {
namespace {
using Atomic = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

struct NativeMissService::Impl {
    kernels::cpu::ExpertPool& pool;
    std::vector<NativeMissSource> sources;
    int n_embd, ranks, max_tokens;
    size_t activation_stride = 0;
    uint8_t* mapped = nullptr;
    uint8_t* alias = nullptr;
    NativeMissPacket* packet = nullptr;
    float* norm = nullptr;
    float* results = nullptr;
    NativeRouteLedger* ledger=nullptr;
    size_t ledger_offset=0;
    std::vector<uint8_t> activation;
    std::array<kernels::cpu::ExpertJobMulti, 16> jobs{};
    std::atomic<bool> stopping{false};
    std::thread worker;
    mutable std::mutex stats_mutex;
    NativeMissStats stats;

    Impl(kernels::cpu::ExpertPool& p, std::vector<NativeMissSource> s, int width, int k, int tokens,bool record_routes)
        : pool(p), sources(std::move(s)), n_embd(width), ranks(k), max_tokens(tokens) {
        if (width < 1 || k < 1 || k > 16 || tokens < 1 || tokens > 2 || k * tokens > 16 || sources.empty())
            throw std::invalid_argument("invalid native miss service geometry");
        if(record_routes && (sources.size()>64 || k!=8 || tokens!=1))
            throw std::invalid_argument("invalid native route ledger geometry");
        size_t act_bytes = 0;
        for (const auto& source : sources) {
            if (!source.gate || !source.up || !source.down || source.n_expert < k ||
                source.format.n_embd != width || !source.format.act_bytes)
                throw std::invalid_argument("invalid native miss source view");
            if(source.format.gu_type==40) {
                if(source.format.d_type!=40 || !source.gate_scales || !source.up_scales || !source.down_scales)
                    throw std::invalid_argument("NVFP4 miss source requires three immutable scale arrays");
                for(int i=0;i<source.n_expert;++i)for(float s:{source.gate_scales[i],source.up_scales[i],source.down_scales[i]})
                    if(!(s>0)||!std::isfinite(s))throw std::invalid_argument("invalid NVFP4 miss global scale");
            }
            if (source.format.act_bytes > act_bytes) act_bytes = source.format.act_bytes;
        }
        activation_stride = act_bytes;
        activation.resize(act_bytes * size_t(max_tokens));
        ledger_offset=sizeof(NativeMissPacket)+size_t(width)*max_tokens*(k+1)*sizeof(float);
        ledger_offset=(ledger_offset+63)&~size_t(63);
        const size_t bytes=ledger_offset+(record_routes?sizeof(NativeRouteLedger):0);
        check(cudaHostAlloc(reinterpret_cast<void**>(&mapped), bytes, cudaHostAllocMapped));
        try { check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&alias), mapped, 0)); }
        catch (...) { cudaFreeHost(mapped); mapped = nullptr; throw; }
        std::memset(mapped, 0, bytes);
        packet = new (mapped) NativeMissPacket{};
        norm = reinterpret_cast<float*>(mapped + sizeof(NativeMissPacket));
        results = norm + size_t(width) * max_tokens;
        if(record_routes)ledger=new(mapped+ledger_offset)NativeRouteLedger{};
    }
    ~Impl() { stop(); if (mapped) cudaFreeHost(mapped); }
    void stop() {
        stopping.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
    }
    void serve(uint32_t epoch,bool pickup_spin) {
        native_miss_pickup::IdleWait<> idle(pickup_spin);
        const auto cores = kernels::cpu::physical_cores(false);
        const auto affinity = kernels::cpu::pin_current_thread(cores.empty() ? -1 : cores[0]);
        Atomic request(packet->request_seq), ack(packet->ack_seq), active(packet->active_epoch);
        uint32_t seen = 0;
        while (!stopping.load(std::memory_order_acquire)) {
            const uint32_t seq = request.load(cuda::memory_order_acquire);
            if (!seq || seq == seen) { idle.idle(); continue; }
            idle.observed();
            seen = seq;
            if (active.load(cuda::memory_order_acquire) != epoch) break;
            const uint32_t mask = packet->mask;
            const uint32_t layer = packet->layer;
            int count = 0;
            packet->status = 0;
            NativeMissStats delta;
            try {
                const int entries = int(packet->ranks);
                const int tokens = entries / ranks;
                if (layer >= sources.size() || entries % ranks || tokens < 1 ||
                    tokens > max_tokens || !mask || (mask >> entries)) throw std::runtime_error("invalid miss request");
                const auto& source = sources[layer];
                const auto& fmt = source.format;
                for (int t = 0; t < tokens; ++t)
                    kernels::cpu::native_quant_act(fmt, norm + size_t(t) * n_embd,
                        activation.data() + size_t(t) * activation_stride);
                int missing = 0;
                for (int rank = 0; rank < entries; ++rank) {
                    if (!(mask & (1u << rank))) continue;
                    const int id = packet->ids[rank];
                    if (id < 0 || id >= source.n_expert) throw std::runtime_error("invalid miss expert");
                    const auto* gate = source.gate + size_t(id) * fmt.up_off;
                    int index = tokens == 1 ? count : 0;
                    for (; index < count && jobs[size_t(index)].gate != gate; ++index) {}
                    if (index == count) {
                        auto& job = jobs[size_t(count++)];
                        job = {}; job.gate = gate;
                        job.up = source.up + size_t(id) * fmt.up_off;
                        job.down = source.down + size_t(id) * (fmt.bytes - fmt.down_off);
                        job.gate_scale=source.gate_scales?source.gate_scales[id]:1.0f;
                        job.up_scale=source.up_scales?source.up_scales[id]:1.0f;
                        job.down_scale=source.down_scales?source.down_scales[id]:1.0f;
                    }
                    auto& job = jobs[size_t(index)];
                    if (job.nt >= tokens) throw std::runtime_error("duplicate miss in one row");
                    job.nact[job.nt] = activation.data() + size_t(rank / ranks) * activation_stride;
                    job.out[job.nt++] = results + size_t(rank) * n_embd;
                    ++missing;
                }
                delta.jobs = count;
                for (int i = 0; i < count; ++i) delta.paired_jobs += jobs[size_t(i)].nt == 2;
                const double gu_before = pool.ms_multi_gu, quant_before = pool.ms_multi_q;
                const double down_before = pool.ms_multi_down;
                const auto before = std::chrono::steady_clock::now();
                pool.run_split_multi_native(fmt, jobs.data(), count);
                delta.pool_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - before).count();
                delta.gu_ms = pool.ms_multi_gu - gu_before;
                delta.quant_ms = pool.ms_multi_q - quant_before;
                delta.down_ms = pool.ms_multi_down - down_before;
                delta.requests = 1; delta.experts = uint64_t(missing);
                delta.expert_bytes = uint64_t(count) * fmt.bytes;
                delta.norm_bytes = size_t(tokens) * n_embd * sizeof(float);
                delta.id_bytes = size_t(entries) * sizeof(int32_t);
                delta.result_bytes = size_t(missing) * n_embd * sizeof(float);
            } catch (...) { packet->status = 1; }
            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.requests += delta.requests; stats.experts += delta.experts;
                stats.expert_bytes += delta.expert_bytes;
                stats.norm_bytes += delta.norm_bytes; stats.id_bytes += delta.id_bytes;
                stats.result_bytes += delta.result_bytes; stats.pool_ms += delta.pool_ms;
                stats.jobs += delta.jobs; stats.paired_jobs += delta.paired_jobs;
                stats.gu_ms += delta.gu_ms; stats.quant_ms += delta.quant_ms; stats.down_ms += delta.down_ms;
            }
            // Publish only after results and the mutex-protected snapshot are complete.
            ack.store(seq, cuda::memory_order_release);
        }
        kernels::cpu::restore_thread_affinity(affinity);
    }
};

NativeMissService::NativeMissService(kernels::cpu::ExpertPool& pool,
    std::vector<NativeMissSource> sources, int n_embd, int ranks, int max_tokens,bool route_ledger)
    : impl_(std::make_unique<Impl>(pool, std::move(sources), n_embd, ranks, max_tokens,route_ledger)) {}
NativeMissService::~NativeMissService() = default;
void NativeMissService::stop() { impl_->stop(); }
void NativeMissService::start(uint32_t epoch) {
    if (!epoch) throw std::invalid_argument("miss epoch must be nonzero");
    const bool pickup_spin=native_miss_pickup::enabled(); // Parent thread: invalid flags never escape a worker.
    impl_->stop();
    if(impl_->ledger)*impl_->ledger=NativeRouteLedger{};
    Atomic(impl_->packet->active_epoch).store(epoch, cuda::memory_order_release);
    Atomic(impl_->packet->request_seq).store(0, cuda::memory_order_release);
    Atomic(impl_->packet->ack_seq).store(0, cuda::memory_order_release);
    { std::lock_guard<std::mutex> lock(impl_->stats_mutex); impl_->stats = {}; }
    impl_->stopping.store(false, std::memory_order_release);
    impl_->worker = std::thread([this, epoch, pickup_spin] { impl_->serve(epoch,pickup_spin); });
}
NativeMissArgs NativeMissService::device_args() const {
    NativeMissArgs args;
    args.packet = reinterpret_cast<NativeMissPacket*>(impl_->alias);
    args.norm_host_alias = reinterpret_cast<float*>(impl_->alias + sizeof(NativeMissPacket));
    args.result_host_alias = args.norm_host_alias + size_t(impl_->n_embd) * impl_->max_tokens;
    args.n_embd = impl_->n_embd; args.ranks = impl_->ranks;
    if(impl_->ledger)args.ledger=reinterpret_cast<NativeRouteLedger*>(impl_->alias+impl_->ledger_offset);
    return args;
}
NativeMissStats NativeMissService::snapshot() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return impl_->stats;
}
const NativeRouteLedger* NativeMissService::route_ledger() const { return impl_->ledger; }

} // namespace strata::core
