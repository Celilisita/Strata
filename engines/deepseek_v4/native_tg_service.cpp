#include "device_cache.hpp"
#include "native_cache_plan.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "mtp_stage_metrics.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace strata::deepseek_v4 {
namespace {
std::atomic<uint64_t> mtp_cpu_jobs{0}, mtp_cpu_compute_ns{0};
uint64_t now_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
}
MtpCpuServiceSnapshot mtp_cpu_service_snapshot() noexcept {
    return {mtp_cpu_jobs.load(std::memory_order_relaxed), mtp_cpu_compute_ns.load(std::memory_order_relaxed)};
}
void DeviceExpertCache::cpu_completed(void* data, bool error) {
    auto& self = *static_cast<DeviceExpertCache*>(data);
    // Capture before release publication: the GPU may immediately publish the
    // next layer after observing this completion.
    auto* const packet = self.dma_->packet();
    const auto ticket = self.cpu_ticket_;
    const auto count = self.cpu_count_;
    const auto elapsed = now_ns() - self.cpu_started_ns_;
    self.cpu_ns_.fetch_add(elapsed, std::memory_order_relaxed);
    if(packet->layer==43){mtp_cpu_jobs.fetch_add(1,std::memory_order_relaxed);
        mtp_cpu_compute_ns.fetch_add(elapsed,std::memory_order_relaxed);}
    self.cpu_sample_ns_[count].store(elapsed, std::memory_order_release);
    self.native_complete_(packet, ticket, error);
}
void DeviceExpertCache::service_native(int32_t layer, const std::array<int32_t, 6>& ids, uint32_t ticket) {
    if (!cpu_ || !native_complete_ || layer < 0 || layer >= int(model_.layers.size()))
        throw std::runtime_error("native TG CPU service unavailable");
    auto* packet = dma_->packet();
    const bool cpu_only = packet->native_request == 2;
    std::array<int32_t, 256> available{};
    std::copy_n(static_cast<const int32_t*>(originals_[layer]->data), 256, available.begin());
    for (size_t rank = 0; rank < ids.size(); ++rank)
        if (ids[rank] < 0 || ids[rank] >= 256 ||
            std::find(ids.begin(), ids.begin() + rank, ids[rank]) != ids.begin() + rank)
            throw std::runtime_error("native TG invalid or duplicate original expert route");
    std::array<int32_t, 6> reusable{};
    if (!cpu_only) {
        cache_.strata_native_slots(layer, ids.data(), reusable.data());
        for (int rank = 0; rank < 6; ++rank) available[ids[rank]] = reusable[rank];
    }
    for (size_t n = 1; n <= 6; ++n) {
        const auto sample = cpu_sample_ns_[n].exchange(0, std::memory_order_acq_rel);
        if (sample) native_policy_.cpu_ns[n] = .9 * native_policy_.cpu_ns[n] + .1 * double(sample);
    }
    if (packet->gpu_slots != native_slots_) throw std::runtime_error("native TG GPU slot destination differs from its owned storage");
    const uint32_t mask = cpu_only ? packet->cpu_mask :
        native_policy_.choose(ids, available.data(), frequency_ ? frequency_host_.data() : nullptr, layer);
    if (cpu_only) {
        uint32_t missing = 0;
        for (size_t rank = 0; rank < 6; ++rank) if (available[ids[rank]] < 0) missing |= 1u << rank;
        if (!mask || mask != missing) throw std::runtime_error("CPU-only native TG miss mask differs from committed residency");
    }
    packet->cpu_mask = mask;
    packet->cpu_error = 0;
    std::array<int32_t, 6> cpu_ids{};
    size_t count = 0;
    for (size_t rank = 0; rank < 6; ++rank)
        if (mask & (1u << rank)) cpu_ids[count++] = ids[rank];
    if (count) {
        const auto& source = model_.layers[layer];
        if (packet->clamp != model_.hparams.swiglu_clamp_exp[layer])
            throw std::runtime_error("native TG activation clamp differs from the model");
        cpu_ticket_ = ticket; cpu_count_ = count; cpu_started_ns_ = now_ns();
        const auto bank=staging_&&staging_->packed_store()?staging_->packed_store()->cpu_bank(source.ffn_gate_exps,packet->clamp):
            Mxfp4CpuBank{source.ffn_gate_exps,source.ffn_up_exps,source.ffn_down_exps,packet->clamp};
        cpu_->submit(bank,
            packet->activation, cpu_ids.data(), count, packet->contribution, cpu_completed, this);
        cpu_jobs_.fetch_add(1, std::memory_order_relaxed);
        cpu_routes_.fetch_add(count, std::memory_order_relaxed);
        cpu_bytes_.fetch_add(count * 13369344ull, std::memory_order_relaxed);
    }
    // CPU-only publication already exposes ready GPU hits. The CPU completion
    // release is the only dependency; no cache mutation, CUDA call or DMA.
    if (cpu_only) return;
    const auto dma_started = now_ns();
    std::array<int32_t, 6> resolved{};
    const auto fills = cache_.strata_fill_native(layer, ids.data(), 63u & ~mask, ticket, resolved.data(), upload, this, &planning_ns_);
    if (fills.permanent_fills) {
        const auto project_started = now_ns();
        project_directory();
        projection_ns_ += now_ns() - project_started;
    }
    uint32_t order_seen = 0;
    for (size_t gpu_rank = 0; gpu_rank < 6; ++gpu_rank) {
        const int rank = packet->gpu_order[gpu_rank];
        if (rank < 0 || rank >= 6 || (order_seen & (1u << rank)))
            throw std::runtime_error("native TG GPU rank order invalid");
        order_seen |= 1u << rank;
        packet->reply_slots[gpu_rank] = resolved[rank];
        if (!(mask & (1u << rank)) && packet->reply_slots[gpu_rank] < 0)
            throw std::runtime_error("native TG GPU admission did not publish a valid resident slot");
    }
    // Directory commit already drains every permanent/transient payload in
    // this request. Pure CPU or reused GPU routes write no payload at all.
    // Only a transient-only fill still needs a copy completion fence.
    dma_->publish_native_slots(fills.bytes && !fills.permanent_fills);
    if (fills.bytes) {
        const double sample = double(now_ns() - dma_started) / (fills.bytes / 13369344.0);
        // Exclude page-fault outliers from the steady-state scheduling model.
        if (sample >= 100000 && sample <= 5000000)
            native_policy_.dma_ns_per_expert = .9 * native_policy_.dma_ns_per_expert + .1 * sample;
    }
    // Worker publishes the DMA reply after this returns. CPU has a separate
    // completion ticket and may still be computing while resident GPU work runs.
}
void DeviceExpertCache::service_pair(int32_t layer,uint32_t ticket) {
    if(!cpu_||!native_complete_||layer<0||layer>=int(model_.layers.size()))throw std::runtime_error("native pair layer invalid");
    auto* p=dma_->packet();if(p->gpu_slots!=native_slots_||p->native_request!=3)throw std::runtime_error("native pair packet storage invalid");
    const auto* directory=static_cast<const int32_t*>(originals_[layer]->data);uint32_t expected=0;
    for(int token=0;token<2;++token)for(int rank=0;rank<6;++rank){int r=token*6+rank,id=p->ids[r];
        if(id<0||id>=256||std::find(p->ids+token*6,p->ids+r,id)!=p->ids+r)throw std::runtime_error("native pair expert ID invalid");
        if(directory[id]<0)expected|=1u<<r;}
    if(!expected||expected!=p->cpu_mask)throw std::runtime_error("native pair miss mask differs from committed directory");
    const auto& l=model_.layers[layer];if(p->clamp!=model_.hparams.swiglu_clamp_exp[layer])throw std::runtime_error("native pair clamp differs");
    const auto bank=staging_&&staging_->packed_store()?staging_->packed_store()->cpu_bank(l.ffn_gate_exps,p->clamp):Mxfp4CpuBank{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps,p->clamp};
    const uint32_t masks[]{expected&63u,(expected>>6)&63u};auto started=now_ns();
    const auto count=cpu_->compute_pair(bank,p->activation,p->ids,masks,p->contribution);
    for(int r=0;r<12;++r)if(expected&(1u<<r))for(int i=0;i<4096;++i)if(!std::isfinite(p->contribution[r*4096+i]))throw std::runtime_error("native pair contribution nonfinite");
    cpu_ns_.fetch_add(now_ns()-started,std::memory_order_relaxed);cpu_jobs_.fetch_add(1,std::memory_order_relaxed);
    if(layer==43){mtp_cpu_jobs.fetch_add(1,std::memory_order_relaxed);
        mtp_cpu_compute_ns.fetch_add(now_ns()-started,std::memory_order_relaxed);}
    cpu_routes_.fetch_add(count,std::memory_order_relaxed);cpu_bytes_.fetch_add(count*13369344ull,std::memory_order_relaxed);
    native_complete_(p,ticket,false);
}

}
