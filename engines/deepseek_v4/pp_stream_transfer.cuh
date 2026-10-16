#pragma once
#include "pp_transport.hpp"
#include "pp_stream_metrics.hpp"
#include "pp_stream_edges.cuh"
#include "ggml-cuda/common.cuh"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
template<class Window>
void pp_upload_rows(const char* source, size_t width, size_t pitch, size_t offset,
    int first, int count, bool fallback_registered, Window& window, cudaStream_t stream,
    PpTransportQuery query, void* owner, const PpStreamEdges* edges, PpStreamMetrics& metrics) {
    constexpr size_t slab_bytes = 16ull << 20;
    size_t consumed = size_t(first) * width;
    const size_t end = consumed + size_t(count) * width;
    while (consumed < end) {
        auto segment = query ? query(source + consumed, end - consumed, owner) :
                               ExpertSourcePin::Segment{end - consumed, fallback_registered};
        // Zero bytes means this projection does not own the source. The probe's
        // separately pinned transport fixture uses the bounded fallback check.
        if (!segment.bytes) segment = {end - consumed, fallback_registered};
        if (segment.bytes > end - consumed) throw std::runtime_error("PP source segment exceeds requested bytes");
        const size_t row = consumed / width, column = consumed % width;
        size_t bytes = std::min(segment.bytes, width - column), rows = 1;
        if (!column && segment.bytes >= width) {
            rows = segment.bytes / width;
            if (!segment.registered) rows = std::min(rows, slab_bytes / width);
            if (!rows) throw std::runtime_error("PP row exceeds bounded staging slab");
            bytes = rows * width;
        }
        const void* input = source + consumed;
        const void* edge = !segment.registered&&edges ? edges->lookup(source+consumed,bytes) : nullptr;
        bool window_staged = false;
        size_t stage_slot = 0;
        if (!segment.registered) {
            metrics.staged_bytes += bytes;
            if(edge) input=edge;
            else {
                const char* setting=std::getenv("STRATA_DSV4_PP_STAGING_SLABS");
                const size_t slots=setting&&std::strcmp(setting,"1")==0?1:3;
                stage_slot=window.staging_next++%slots;
                auto& slab=window.staging[stage_slot];
                slab.reserve(slab_bytes);
                if (window.staged_recorded[stage_slot]) {
                    const auto wait = std::chrono::steady_clock::now();
                    CUDA_CHECK(cudaEventSynchronize(window.staged[stage_slot]));
                    metrics.staging_wait_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - wait).count());
                }
                const auto start = std::chrono::steady_clock::now();
                std::memcpy(slab.p, input, bytes);
                metrics.source_cpu_copy_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count());
                input = slab.p;
                window_staged = true;
            }
        }
        void* destination = window.device.template as<char>(offset + row * pitch + column);
        auto copy=[&]{if(rows==1||pitch==width)CUDA_CHECK(cudaMemcpyAsync(destination,input,bytes,cudaMemcpyHostToDevice,stream));
            else CUDA_CHECK(cudaMemcpy2DAsync(destination,pitch,input,width,width,rows,cudaMemcpyHostToDevice,stream));};
        if(window.trace)window.trace->copy(copy);else copy();
        if (window_staged) {
            CUDA_CHECK(cudaEventRecord(window.staged[stage_slot], stream));
            window.staged_recorded[stage_slot] = true;
        }
        metrics.payload_bytes += bytes;
        ++metrics.dma_calls;
        consumed += bytes;
    }
}
}
