#include "expert_staging.hpp"
#include "pp_transport.hpp"
#include "pp_stream_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::deepseek_v4 {

namespace {
using StagingClock = std::chrono::steady_clock;

void add_wall_time(uint64_t& counter, StagingClock::time_point start,
                   StagingClock::time_point end) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (elapsed < 0) throw std::runtime_error("expert staging wall clock moved backwards");
    const uint64_t ns = static_cast<uint64_t>(elapsed);
    if (ns > std::numeric_limits<uint64_t>::max() - counter)
        throw std::runtime_error("expert staging wall time exceeds its fixed counter range");
    counter += ns;
}
} // namespace

ExpertStaging::ExpertStaging(ggml_backend_t backend) : backend_(backend) {
    if (!backend_) throw std::runtime_error("expert staging requires a selected CUDA backend");
    const auto device = ggml_backend_get_device(backend_);
    const auto registration = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    const char* registration_name = registration ? ggml_backend_reg_name(registration) : nullptr;
    if (!registration_name || std::strcmp(registration_name, "CUDA") != 0)
        throw std::runtime_error("expert staging requires the canonical CUDA backend");

    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(device, &properties);
    if (!properties.caps.async || !properties.caps.host_buffer || !properties.caps.events)
        throw std::runtime_error("expert staging requires pinned host buffers and backend events");
    const auto host_type = ggml_backend_dev_host_buffer_type(device);
    const char* host_name = host_type ? ggml_backend_buft_name(host_type) : nullptr;
    if (!host_name || std::strcmp(host_name, "CUDA_Host") != 0)
        throw std::runtime_error("expert staging requires the canonical CUDA_Host buffer type");

    try {
        for (auto& slab : slabs_) {
            // The canonical CUDA event constructor selects the requested
            // device before its host allocation is made.
            slab.event = ggml_backend_event_new(device);
            if (!slab.event) throw std::runtime_error("expert staging event allocation failed");
            slab.buffer = ggml_backend_buft_alloc_buffer(host_type, slab_bytes);
            if (!slab.buffer) throw std::runtime_error("expert staging host allocation failed");
            const auto actual_type = ggml_backend_buffer_get_type(slab.buffer);
            const char* actual_name = actual_type ? ggml_backend_buft_name(actual_type) : nullptr;
            if (actual_type != host_type || !actual_name || std::strcmp(actual_name, "CUDA_Host") != 0 ||
                !ggml_backend_buffer_is_host(slab.buffer) ||
                ggml_backend_buffer_get_size(slab.buffer) < slab_bytes)
                throw std::runtime_error("expert staging allocation fell back to pageable memory");
            slab.data = ggml_backend_buffer_get_base(slab.buffer);
            if (!slab.data) throw std::runtime_error("expert staging host address is unavailable");
        }
    } catch (...) {
        release();
        throw;
    }
}

ExpertStaging::~ExpertStaging() noexcept {
    release();
}

void ExpertStaging::release() noexcept {
    if(pp_packed_bound_){bind_pp_packed(backend_,nullptr,nullptr);pp_packed_bound_=false;}
    if (pp_transport_bound_) {
        bind_pp_transport(backend_, nullptr, nullptr);
        pp_transport_bound_ = false;
    }
    source_pin_.reset();
    // Drain both submitted slabs before freeing either source allocation.
    for (auto& slab : slabs_) {
        if (slab.busy && slab.event) {
            ggml_backend_event_synchronize(slab.event);
            slab.busy = false;
        }
    }
    for (auto& slab : slabs_) {
        if (slab.event) ggml_backend_event_free(slab.event);
        if (slab.buffer) ggml_backend_buffer_free(slab.buffer);
        slab = {};
    }
    packed_.reset();
}

void ExpertStaging::pack_sources(const std::vector<ggml_tensor*>& sources,std::shared_ptr<PackedExpertStore>* retained) {
    if(source_pin_||packed_||metrics_.calls)throw std::runtime_error("expert repacking must precede transport");
    if(retained&&*retained){if(!(*retained)->matches(backend_,sources))throw std::runtime_error("retained packed inventory differs");packed_=*retained;}
    else{packed_=std::make_shared<PackedExpertStore>(backend_,sources);if(retained)*retained=packed_;}
    if(pp_stream_enabled()){
        bind_pp_packed(backend_,[](const ggml_tensor* gate,int first,int count,void* owner){
            return static_cast<PackedExpertStore*>(owner)->span(gate,first,count);
        },packed_.get());pp_packed_bound_=true;
    }
}

void ExpertStaging::register_sources(const std::vector<ggml_tensor*>& sources, uint64_t budget,
                                    std::shared_ptr<ExpertSourcePin>* retained) {
    if (source_pin_ || metrics_.calls)
        throw std::runtime_error("expert source pinning must precede the first upload");
    if (!budget) return;
    // Reserve the PP control events/stream before the large source registration.
    // No transport reads are submitted until initialization returns.
    if (pp_stream_enabled() || pp_stream_enabled(true)) {
        bind_pp_transport(backend_, [](const void* p, size_t n, void* owner) {
            return static_cast<ExpertStaging*>(owner)->transport_segment(p, n);
        }, this);
        pp_transport_bound_ = true;
    }
    if (retained && *retained) {
        if (!(*retained)->matches(backend_, sources, budget)) throw std::runtime_error("retained source registration identity differs");
        source_pin_ = *retained;
    } else {
        source_pin_ = std::make_shared<ExpertSourcePin>(backend_, sources, budget);
        if (retained) *retained = source_pin_;
    }
}

void ExpertStaging::upload(ggml_backend_t backend, ggml_tensor* destination,
                           const void* source, std::size_t destination_offset, std::size_t bytes) {
    if (backend != backend_) throw std::runtime_error("expert staging backend differs from its owner");
    if (bytes == 0) return;
    if (!destination || !destination->data || !source)
        throw std::runtime_error("expert staging requires allocated destination and source data");
    const auto buffer = destination->view_src ? destination->view_src->buffer : destination->buffer;
    if (!buffer || ggml_backend_buffer_get_type(buffer) != ggml_backend_get_default_buffer_type(backend_))
        throw std::runtime_error("expert staging destination is not on its selected CUDA backend");
    const std::size_t destination_bytes = ggml_nbytes(destination);
    if (destination_offset > destination_bytes || bytes > destination_bytes - destination_offset)
        throw std::runtime_error("expert staging upload exceeds destination bounds");
    const uint64_t chunks = bytes / slab_bytes + (bytes % slab_bytes != 0);
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (bytes > maximum - metrics_.bytes || chunks > maximum - metrics_.calls ||
        chunks > maximum - metrics_.slab_waits)
        throw std::runtime_error("expert staging metrics exceed their fixed counter range");

    const auto* input = static_cast<const uint8_t*>(source);
    std::size_t consumed = 0;
    while (consumed < bytes) {
        if(packed_){const auto projection=packed_->project(input+consumed,bytes-consumed);
            if(!projection.data||!projection.bytes)throw std::runtime_error("expert upload outside packed inventory");
            ggml_backend_tensor_set_async(backend_,destination,projection.data,destination_offset+consumed,projection.bytes);
            metrics_.bytes+=projection.bytes;++metrics_.calls;metrics_.direct_bytes+=projection.bytes;++metrics_.direct_calls;
            consumed+=projection.bytes;continue;
        }
        const auto segment = source_pin_ ? source_pin_->segment(input + consumed, bytes - consumed) :
                                          ExpertSourcePin::Segment{bytes - consumed, false};
        if (!segment.bytes || metrics_.calls == maximum)
            throw std::runtime_error("expert transport segment or call counter is invalid");
        if (segment.registered) {
            if (segment.bytes > maximum - metrics_.direct_bytes || metrics_.direct_calls == maximum)
                throw std::runtime_error("direct expert transport counter overflows");
            ggml_backend_tensor_set_async(backend_, destination, input + consumed,
                                          destination_offset + consumed, segment.bytes);
            source_pin_->submitted(backend_);
            metrics_.bytes += segment.bytes;
            ++metrics_.calls;
            metrics_.direct_bytes += segment.bytes;
            ++metrics_.direct_calls;
            consumed += segment.bytes;
            continue;
        }
        auto& slab = slabs_[next_slab_];
        if (slab.busy) {
            if (metrics_.slab_waits == maximum)
                throw std::runtime_error("expert slab wait counter overflows");
            const auto wait_start = StagingClock::now();
            ggml_backend_event_synchronize(slab.event);
            const auto wait_end = StagingClock::now();
            slab.busy = false;
            add_wall_time(metrics_.slab_wait_wall_ns, wait_start, wait_end);
            ++metrics_.slab_waits;
        }
        const std::size_t size = std::min(slab_bytes, segment.bytes);
        const auto copy_start = StagingClock::now();
        std::memcpy(slab.data, input + consumed, size);
        metrics_.host_copy_bytes += size;
        const auto copy_end = StagingClock::now();
        add_wall_time(metrics_.host_copy_wall_ns, copy_start, copy_end);
        // Canonical set_async uses this backend's existing compute stream.
        ggml_backend_tensor_set_async(backend_, destination, slab.data, destination_offset + consumed, size);
        ggml_backend_event_record(slab.event, backend_);
        slab.busy = true;
        metrics_.bytes += size;
        ++metrics_.calls;
        consumed += size;
        next_slab_ = (next_slab_ + 1) % slab_count;
    }
}

bool ExpertStaging::dispatch(ggml_backend_t backend, ggml_tensor* destination,
                             const void* source, std::size_t destination_offset,
                             std::size_t bytes, void* user_data) {
    if (!user_data) return false;
    static_cast<ExpertStaging*>(user_data)->upload(backend, destination, source, destination_offset, bytes);
    return true;
}

} // namespace strata::deepseek_v4
