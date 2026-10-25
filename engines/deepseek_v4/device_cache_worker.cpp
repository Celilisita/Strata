#include "device_cache.hpp"
#include "../glm5_next/hybrid_provider.hpp"
#include "../glm5_next/cold_gpu_owner.hpp"
#include "../glm5_next/cold_gpu_protocol.hpp"
#include "../glm5_next/partial_down_host.hpp"
#include "expert_frequency.hpp"
#include "llama-moe-cache.h"
#include "llama-model.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
#include "../glm5_next/hybrid_service.inc"
#include "../glm5_next/hybrid_four_service.inc"
void DeviceExpertCache::read_frequency() {
    dma_->read_frequency(frequency_gpu_->data, frequency_host_);
    frequency_->import_state(frequency_host_.data(), frequency_host_.size());
}
void DeviceExpertCache::project_directory() {
    for (int layer = 0; layer < int(model_.layers.size()); ++layer)
        if(originals_[layer])std::memcpy(directory_host_.data() + layer * experts_, originals_[layer]->data, experts_ * sizeof(int32_t));
    dma_->commit(directory_->data, directory_host_);
    directory_bytes_ += directory_host_.size() * sizeof(int32_t);
}
bool DeviceExpertCache::upload(ggml_backend_t backend, ggml_tensor* destination,
                              const void* source, size_t offset, size_t bytes, void* data) {
    auto& self = *static_cast<DeviceExpertCache*>(data);
    if (backend != self.backend_) throw std::runtime_error("device miss backend differs");
    self.dma_->upload(destination, source, offset, bytes);
    return true;
}
void DeviceExpertCache::worker() {
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        try { dma_->bind_thread(); }
        catch (const std::exception& failure) { error_ = failure.what(); }
        worker_ready_ = true;
    }
    wake_.notify_one();
    auto* packet = dma_->packet();
    uint32_t seen = 0;
    while (!stop_.load(std::memory_order_acquire)) {
        if (!worker_active_.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(wake_mutex_);
            wake_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) ||
                worker_active_.load(std::memory_order_acquire); });
            continue;
        }
        const uint32_t sequence = native_ready_ ? native_ready_(packet) : packet->sequence;
        if (sequence == seen) {
            if(read_ahead_&&cpu_->idle()){std::lock_guard<std::mutex> idle_lock(service_mutex_);
                if(read_ahead_->poll(seen))continue;}
            // Windows rounds short sleeps to scheduler timeouts. Keep TG
            // runnable; inactive PP still waits on the condition variable.
            std::this_thread::yield();
            continue;
        }
        const bool causal_candidate=causal_host_.candidate(sequence);
        const uint64_t causal_notice=causal_candidate?strata::glm5_next::causal_host_now_ns():0;
        std::atomic_thread_fence(std::memory_order_acquire);
        std::lock_guard<std::mutex> lock(service_mutex_);
        const auto started = std::chrono::steady_clock::now();
        auto causal=causal_candidate?causal_host_.begin(sequence,owner_epoch_,packet->owner_epoch,int(packet->layer),
            packet->native_request,packet->routed_count,causal_notice,strata::glm5_next::causal_host_ns(started)):
            strata::glm5_next::CausalHostSample{};
        try {
            if (!error_.empty()) throw std::runtime_error(error_);
            if (packet->owner_epoch != owner_epoch_) throw std::runtime_error("device miss belongs to a stale model/context epoch");
            const int32_t layer = int32_t(packet->layer);
            std::array<int32_t, 6> ids{};
            std::memcpy(ids.data(), packet->ids, sizeof(ids));
            if (frequency_ && packet->native_request != 2 && packet->native_request != 3 && packet->native_request != 5) {
                dma_->read_miss_frequency(frequency_host_);
                frequency_->import_state(frequency_host_.data(), frequency_host_.size());
            }
            if(read_ahead_&&packet->native_request)read_ahead_->observe(*packet);
            if (packet->native_request == 4 || packet->native_request == 5) service_hybrid(layer,causal.active?&causal:nullptr);
            else if (packet->native_request == 3) service_pair(layer, sequence);
            else if (packet->native_request) service_native(layer, ids, sequence);
            else {
                // The requesting compute-stream lookup is paused after the
                // preceding dependent kernels. No earlier slot reader remains.
                // Fill through the one host cache authority, complete payload
                // and directory DMA, then publish the response ticket. The
                // context destructor drains CUDA and joins this worker before
                // releasing any directory, slot, source mapping or staging.
                if(layer < 0 || layer >= int(model_.layers.size()) || !originals_[layer])
                    throw std::runtime_error("device miss key is outside the active model inventory");
                auto& count = last_miss_counts_[layer]; count = 0;
                const auto* slots = static_cast<const int32_t*>(originals_[layer]->data);
                for(int rank=0; rank<top_k_; ++rank) {
                    const int id = packet->ids[rank];
                    if(id<0 || id>=experts_)throw std::runtime_error("device miss expert key invalid");
                    if(slots[id]<0)last_miss_ids_[layer][count++]=id;
                }
                cache_.strata_fill_selected(layer, packet->ids, size_t(top_k_), upload, this, &planning_ns_, frequency_ != nullptr, (1u<<top_k_)-1);
                const auto project_started = std::chrono::steady_clock::now();
                project_directory();
                projection_ns_ += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - project_started).count());
            }
            ++services_;
        } catch (const std::exception& failure) {
            error_ = failure.what();
            packet->error = 1;
            if ((packet->native_request == 2 || packet->native_request == 3) && native_complete_) {
                native_complete_(packet, sequence, true);
            } else if (packet->native_request && native_slots_ && packet->gpu_slots == native_slots_) {
                std::fill_n(packet->reply_slots, 6, -3);
                try { dma_->publish_native_slots(); }
                catch (const std::exception& transport) { error_ += "; native failure publication: "; error_ += transport.what(); }
            }
        }
        const auto elapsed = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
        service_ns_ += elapsed;
        if (elapsed > service_max_ns_) service_max_ns_ = elapsed;
        seen = sequence;
        // Validate before release: the next GPU lookup may overwrite the packet
        // immediately after native_reply_. No packet reads follow that call.
        const bool causal_identity=causal.active&&packet->owner_epoch==causal.owner_epoch&&packet->sequence==causal.ticket&&
            int(packet->layer)==causal.layer&&packet->native_request==5&&packet->routed_count==causal.routes;
        const bool causal_service_error=causal.active&&packet->error;
        std::atomic_thread_fence(std::memory_order_release);
        if (native_reply_) native_reply_(packet, sequence);
        else packet->response = sequence;
        if(causal.active)causal_host_.complete(causal,causal_identity,causal_service_error,strata::glm5_next::causal_host_now_ns());
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}
}
