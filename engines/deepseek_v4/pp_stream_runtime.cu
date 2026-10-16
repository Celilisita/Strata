#include "pp_stream_cuda.cuh"
#include "pp_stream_plan.hpp"
#include "pp_stream_transfer.cuh"
#include "pp_stream_timing.cuh"
#include "pp_packed_upload.cuh"
#include "pp_host_trace.hpp"
#include <utility>
#include "quantize.cuh"
#include "ggml-backend-impl.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace strata::deepseek_v4 {
namespace {
struct Memory {
    void* p = nullptr;
    size_t bytes = 0;
    bool host = false;
    uint64_t alloc_ns = 0, alloc_calls = 0;
    explicit Memory(bool h = false) : host(h) {}
    ~Memory() { clear(); }
    void clear() { if (p) { if (host) cudaFreeHost(p); else cudaFree(p); } p = nullptr; bytes = 0; }
    void reserve(size_t n) {
        if (bytes >= n) return;
        clear();
        const auto start = std::chrono::steady_clock::now();
        const auto status = host ? cudaHostAlloc(&p, n, cudaHostAllocDefault) : cudaMalloc(&p, n);
        if (!host) {
            alloc_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
            ++alloc_calls;
        }
        if (status != cudaSuccess) {
            p = nullptr;
            (void) cudaGetLastError();
            throw std::runtime_error(std::string("PP ") + (host ? "host" : "device") +
                " allocation failed for " + std::to_string(n) + " bytes: " + cudaGetErrorString(status));
        }
        bytes = n;
    }
    template<class T> T* as(size_t offset = 0) { return reinterpret_cast<T*>(static_cast<char*>(p) + offset); }
};
struct Window {
    Memory device;
    std::array<Memory,3> staging;
    PpHostTrace* trace=nullptr;
    cudaEvent_t ready = nullptr, done = nullptr;
    bool ready_recorded = false, done_recorded = false;
    bool packed = false;
    std::array<cudaEvent_t,3> staged{};
    std::array<bool,3> staged_recorded{};
    size_t staging_next = 0;
    Window() {
        for (auto& slab:staging) slab.host=true;
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        for(auto& event:staged)CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
    }
    ~Window() {
        if (ready) cudaEventDestroy(ready);
        if (done) cudaEventDestroy(done);
        for(auto event:staged)if(event)cudaEventDestroy(event);
    }
};
bool pinned(const void* p) {
    cudaPointerAttributes a{};
    const auto status = cudaPointerGetAttributes(&a, p);
    if (status != cudaSuccess) { cudaGetLastError(); return false; }
    return a.type == cudaMemoryTypeHost;
}
bool pinned_span(const void* p, size_t bytes) {
    // Current model registration is one contiguous interval per bank, and a
    // cudaHostAlloc probe bank is one interval. Include both unregistered mmap
    // edge pages and the budget's partial final bank in the staging decision.
    return bytes && pinned(p) && pinned(static_cast<const char*>(p) + bytes - 1);
}
size_t quant_bytes(int rows, int cols, bool fp4) {
    return size_t(rows) * cols * sizeof(block_q8_1_mmq) / (fp4 ? QK_FP4_MMQ : QK8_1_MMQ) +
           128 * sizeof(block_q8_1_mmq);
}
struct Pipeline {
    std::array<Window, 2> windows;
    Memory route_host{true}, map_host{true}, maps, xq, gu, h, hq, down;
    cudaStream_t copy = nullptr;
    int device;
    PpStreamMetrics metrics;
    PpHostTrace trace;
    PpStreamTiming timing;
    PpStreamEdges edges;
    PpTransportQuery source_query = nullptr;
    void* source_owner = nullptr;
    PackedExpertQuery packed_query = nullptr;
    void* packed_owner = nullptr;
    std::array<PpPackedPrefix<Memory>,2> prefixes;
    explicit Pipeline(int d) : device(d) { CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));for(auto& window:windows)window.trace=&trace; }
    ~Pipeline() {
        cudaSetDevice(device);
        if (copy) { cudaStreamSynchronize(copy); cudaStreamDestroy(copy); }
    }
    void enter_tg(cudaStream_t compute) {
        // At a phase boundary, all window readers/copies must finish before
        // their scratch is retired. Keep counters and small host staging.
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaStreamSynchronize(copy));
        for (auto& window : windows) window.device.clear();
        for(auto& prefix:prefixes)prefix.clear();
        maps.clear(); xq.clear(); gu.clear(); h.clear(); hq.clear(); down.clear();
    }
    void enqueue(const PpStreamSources& s, const PpStreamGroup& g, Window& w) {
        w.packed=false;
        if(g.cache_first_slot>=0) {
            // Ready cache rows are consumed directly by canonical MMQ. Hits
            // delete host uploads and the intermediate device weight copy.
            metrics.cache_reused_bytes+=uint64_t(g.experts)*(2*s.gate->nb[2]+s.down->nb[2]);
            CUDA_CHECK(cudaEventRecord(w.ready,copy));w.ready_recorded=true;
            return;
        }
        const size_t half = s.gate->nb[2], gu_stride = 2 * half, d_stride = s.down->nb[2];
        const auto* gate = static_cast<const char*>(s.gate->data) + g.first_expert * half;
        const auto* up = static_cast<const char*>(s.up->data) + g.first_expert * half;
        const auto* dn = static_cast<const char*>(s.down->data) + g.first_expert * d_stride;
        // Waits are issued only after the previous reader has been recorded.
        // Waiting on an unrecorded CUDA event would provide no protection.
        if (w.done_recorded) CUDA_CHECK(cudaStreamWaitEvent(copy, w.done, 0));
        if(packed_query) {
            const PpPrefetchedSpan* prefetched=nullptr;
            for(const auto& prefix:prefixes)if(prefix.span.gate==s.gate)prefetched=&prefix.span;
            pp_packed_upload(s,g,packed_query(s.gate,g.first_expert,g.experts,packed_owner),w.device.p,copy,metrics,prefetched);
            w.packed=true;CUDA_CHECK(cudaEventRecord(w.ready,copy));w.ready_recorded=true;return;
        }
        const size_t down_offset = size_t(s.window_experts) * gu_stride;
        const int32_t* slots = s.cache ? static_cast<const int32_t*>(s.cache->directory->data) : nullptr;
        auto matrix = [&](const char* src, size_t width, size_t dst_pitch, size_t offset, int role) {
            auto submit = [&](int first, int count, bool registered) {
                const char* setting = std::getenv("STRATA_DSV4_PP_SOURCE_RANGES");
                const bool ranges = source_query && !(setting && std::strcmp(setting, "0") == 0);
                pp_upload_rows(src, width, dst_pitch, offset, first, count, registered,
                    w, copy, ranges ? source_query : nullptr, source_owner, &edges, metrics);
            };
            const char* setting = std::getenv("STRATA_DSV4_PP_SOURCE_RANGES");
            const bool ranges = source_query && !(setting && std::strcmp(setting, "0") == 0);
            const auto owned = ranges ? source_query(src, width * g.experts, source_owner) : ExpertSourcePin::Segment{};
            const bool projected = owned.bytes != 0;
            if (!slots && (projected || pinned_span(src, width * g.experts))) { submit(0, g.experts, true); return; }
            for (int first = 0; first < g.experts;) {
                const int slot = slots ? slots[g.first_expert + first] : -1;
                if (slot >= 0) {
                    const auto* bank = s.cache->bank[role];
                    if (slot >= bank->ne[2]) throw std::runtime_error("PP cache slot outside owned bank");
                    int end = first + 1;
                    while (end < g.experts && slots[g.first_expert + end] == slot + end - first) ++end;
                    const char* input = static_cast<const char*>(bank->data) + size_t(slot) * width;
                    char* dst = w.device.as<char>(offset + size_t(first) * dst_pitch);
                    trace.copy([&]{CUDA_CHECK(cudaMemcpy2DAsync(dst, dst_pitch, input, width, width,
                        end - first, cudaMemcpyDeviceToDevice, copy));});
                    metrics.cache_reused_bytes += size_t(end - first) * width;
                    ++metrics.cache_copy_calls;
                    first = end; continue;
                }
                const bool registered = projected || pinned_span(src + size_t(first) * width, width);
                int end = first + 1;
                while (end < g.experts && (!slots || slots[g.first_expert + end] < 0) &&
                    (projected || pinned_span(src + size_t(end) * width, width)) == registered) ++end;
                submit(first, end - first, registered); first = end;
            }
        };
        matrix(gate, half, gu_stride, 0, 0);
        matrix(up, half, gu_stride, half, 1);
        matrix(dn, d_stride, d_stride, down_offset, 2);
        CUDA_CHECK(cudaEventRecord(w.ready, copy)); w.ready_recorded = true;
    }
    void run(ggml_backend_cuda_context& ctx, ggml_tensor* out) {
        const auto s = pp_stream_sources(out);
        const auto payload_before=metrics.payload_bytes,reused_before=metrics.cache_reused_bytes;
        auto product=[&](auto&&... args){trace.product([&]{pp_stream_product(std::forward<decltype(args)>(args)...);});};
        for (const auto* bank : {s.gate, s.up, s.down})
            if (!bank || !bank->data || !bank->buffer || !ggml_backend_buffer_is_host(bank->buffer))
                throw std::runtime_error("PP execution requires live immutable host banks; reserve-only metadata cannot execute");
        if (s.cache) {
            const std::array<const ggml_tensor*,3> originals{s.gate,s.up,s.down};
            if (s.cache->source != originals || !s.cache->directory || !s.cache->directory->data ||
                s.cache->directory->type != GGML_TYPE_I32 || ggml_nelements(s.cache->directory) != 256)
                throw std::runtime_error("PP cache source/key projection differs");
            for (int role=0;role<3;++role) {
                const auto* bank=s.cache->bank[role];
                if (!bank || !bank->data || bank->type != GGML_TYPE_MXFP4 ||
                    bank->nb[2] != originals[role]->nb[2]) throw std::runtime_error("PP cached payload layout differs");
            }
        }
        auto* x = out->src[0]; auto* ids = out->src[1]; auto* weights = out->src[2];
        if (!is_pp_stream(out) || !ggml_is_contiguous(x) || !ggml_is_contiguous(ids) ||
            !ggml_is_contiguous(weights) || !ggml_is_contiguous(out))
            throw std::runtime_error("PP streaming inputs must be contiguous");
        const int tokens = int(x->ne[1]), pairs = tokens * 6;
        const auto stream = ctx.stream();
        const char* precision = std::getenv("GGML_CUDA_MMQ_PREC");
        const bool fp4 = blackwell_mma_available(ggml_cuda_info().devices[device].cc) &&
                         !(precision && std::strcmp(precision, "q8") == 0);
        route_host.reserve(pairs * sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(route_host.p, ids->data, pairs * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++metrics.route_host_reads;
        timing.collect(metrics);
        const auto plan = make_pp_stream_plan(route_host.as<int32_t>(), tokens, 6, 256,
                                              s.window_experts, 2 * s.gate->nb[2] + s.down->nb[2],
                                              s.cache?static_cast<const int32_t*>(s.cache->directory->data):nullptr);
        if(s.cache&&s.cache->prefill_counts&&s.cache->prefill_tokens){
            std::fill_n(s.cache->prefill_counts,256,0u);
            for(int pair=0;pair<pairs;++pair)++s.cache->prefill_counts[route_host.as<int32_t>()[pair]];
            *s.cache->prefill_tokens=tokens;
        }
        timing.begin(plan.groups.size(), stream, metrics);
        const char* edge_setting=std::getenv("STRATA_DSV4_PP_FIXED_EDGES");
        const char* range_setting=std::getenv("STRATA_DSV4_PP_SOURCE_RANGES");
        const bool fixed_edges=!(edge_setting&&std::strcmp(edge_setting,"0")==0)&&
            !(range_setting&&std::strcmp(range_setting,"0")==0);
        // The route read synchronized the previous layer's final readers and
        // all its window copies. Edge storage may now be reused for this layer.
        edges.prepare({s.gate,s.up,s.down},fixed_edges?source_query:nullptr,source_owner,metrics);
        const size_t window_bytes = size_t(s.window_experts) * (2 * s.gate->nb[2] + s.down->nb[2]);
        if(s.cache) for(const auto& g:plan.groups) if(g.cache_first_slot>=0)
            for(const auto* bank:s.cache->bank)
                if(int64_t(g.cache_first_slot)+g.experts>bank->ne[2])
                    throw std::runtime_error("PP direct cache interval exceeds its owned bank");
        if(std::any_of(plan.groups.begin(),plan.groups.end(),[](const auto& g){return g.cache_first_slot<0;}))
            for (auto& w : windows) w.device.reserve(window_bytes);
        // High-water allocations are reused across layers; no per-expert alloc.
        const int r = plan.max_group_rows;
        xq.reserve(quant_bytes(pairs, 4096, fp4));
        gu.reserve(size_t(r) * 4096 * sizeof(float)); h.reserve(size_t(r) * 2048 * sizeof(float));
        hq.reserve(quant_bytes(r, 2048, fp4)); down.reserve(size_t(r) * 4096 * sizeof(float));
        const size_t bounds_start = 2 * size_t(pairs) + r + 127;
        size_t table_ints=bounds_start;
        for(const auto& g:plan.groups)table_ints+=17*g.bounds.size();
        map_host.reserve(table_ints * sizeof(int32_t)); maps.reserve(table_ints * sizeof(int32_t));
        auto* host = map_host.as<int32_t>();
        std::copy(plan.inverse.begin(), plan.inverse.end(), host);
        std::copy(plan.sorted_pairs.begin(), plan.sorted_pairs.end(), host + pairs);
        size_t pos = bounds_start;
        std::vector<size_t> bound_offsets,ragged_offsets;
        for (const auto& g : plan.groups) {
            bound_offsets.push_back(pos);
            std::copy(g.bounds.begin(), g.bounds.end(), host + pos); pos += g.bounds.size();
            ragged_offsets.push_back(pos);
            for(int width=8;width<=128;width+=8) {
                host[pos]=0;
                for(int expert=0;expert<g.experts;++expert)
                    host[pos+expert+1]=host[pos+expert]+(g.bounds[expert+1]-g.bounds[expert]+width-1)/width;
                pos+=g.bounds.size();
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(maps.p, map_host.p, pos * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        auto* inverse = maps.as<int32_t>(); auto* sorted = inverse + pairs; auto* identity = sorted + pairs;
        pp_stream_iota(identity, r + 127, stream);
        CUDA_CHECK(cudaMemsetAsync(xq.as<char>(quant_bytes(pairs, 4096, fp4) - 128 * sizeof(block_q8_1_mmq)),
                                   0, 128 * sizeof(block_q8_1_mmq), stream));
        if (fp4) quantize_scatter_mmq_fp4_cuda(static_cast<const float*>(x->data), inverse, xq.p, nullptr,
            GGML_TYPE_MXFP4, true, 4096, 4096, 4096, tokens, pairs, 6, stream);
        else quantize_scatter_mmq_q8_1_cuda(static_cast<const float*>(x->data), inverse, xq.p,
            GGML_TYPE_MXFP4, 4096, 4096, 4096, tokens, pairs, 6, stream);
        CUDA_CHECK(cudaMemsetAsync(out->data, 0, ggml_nbytes(out), stream));
        for (size_t i = 0; i < std::min<size_t>(2, plan.groups.size()); ++i) enqueue(s, plan.groups[i], windows[i]);
        const size_t down_offset = size_t(s.window_experts) * 2 * s.gate->nb[2];
        const char* ragged_setting=std::getenv("STRATA_DSV4_PP_RAGGED_MMQ");
        const bool ragged=!ragged_setting||std::strcmp(ragged_setting,"0")!=0;
        for (size_t i = 0; i < plan.groups.size(); ++i) {
            const auto& g = plan.groups[i]; auto& w = windows[i % 2];
            timing.waiting(i, stream);
            CUDA_CHECK(cudaStreamWaitEvent(stream, w.ready, 0));
            timing.ready(i, stream);
            const auto* bounds = maps.as<int32_t>() + bound_offsets[i];
            const auto* tile_prefix=ragged?maps.as<int32_t>()+ragged_offsets[i]:nullptr;
            const auto* host_bounds=ragged?g.bounds.data():nullptr;
            // MMQ activation blocks are K-major: [K-block][compact row].
            // A group advances one block per row and retains the full route
            // count as its K-block stride, rather than using a row-major offset.
            const void* down_weights=nullptr;
            if(g.cache_first_slot>=0) {
                const auto base=[&](int role){return static_cast<const char*>(s.cache->bank[role]->data)+
                    size_t(g.cache_first_slot)*s.cache->bank[role]->nb[2];};
                product(ctx,base(0),xq.as<char>(g.first_row*sizeof(block_q8_1_mmq)),bounds,identity,
                    gu.as<float>(),g.experts,g.rows,pairs,g.max_rows,4096,2048,s.gate->nb[2],fp4,stream,4096,tile_prefix,host_bounds);
                product(ctx,base(1),xq.as<char>(g.first_row*sizeof(block_q8_1_mmq)),bounds,identity,
                    gu.as<float>()+2048,g.experts,g.rows,pairs,g.max_rows,4096,2048,s.up->nb[2],fp4,stream,4096,tile_prefix,host_bounds);
                down_weights=base(2);++metrics.grouped_products;
            } else {
                const size_t stride=w.packed?2*s.gate->nb[2]+s.down->nb[2]:2*s.gate->nb[2];
                down_weights=w.device.as<char>(w.packed?2*s.gate->nb[2]:down_offset);
                product(ctx, w.device.p, xq.as<char>(g.first_row * sizeof(block_q8_1_mmq)), bounds, identity,
                    gu.as<float>(), g.experts, g.rows, pairs, g.max_rows, 4096, 4096, stride, fp4, stream,0,tile_prefix,host_bounds);
            }
            pp_stream_swiglu(gu.as<float>(), h.as<float>(), g.rows, s.clamp, stream);
            CUDA_CHECK(cudaMemsetAsync(hq.as<char>(quant_bytes(g.rows, 2048, fp4) - 128 * sizeof(block_q8_1_mmq)),
                                       0, 128 * sizeof(block_q8_1_mmq), stream));
            if (fp4) quantize_mmq_fp4_cuda(h.as<float>(), nullptr, hq.p, nullptr, GGML_TYPE_MXFP4,
                true, 2048, 2048, int64_t(g.rows) * 2048, int64_t(g.rows) * 2048, 2048, g.rows, 1, 1, stream);
            else quantize_mmq_q8_1_cuda(h.as<float>(), nullptr, hq.p, GGML_TYPE_MXFP4,
                2048, 2048, int64_t(g.rows) * 2048, int64_t(g.rows) * 2048, 2048, g.rows, 1, 1, stream);
            product(ctx, down_weights, hq.p, bounds, identity,
                down.as<float>(), g.experts, g.rows, g.rows, g.max_rows, 2048, 4096,
                w.packed?2*s.gate->nb[2]+s.down->nb[2]:s.down->nb[2], fp4, stream,0,tile_prefix,host_bounds);
            pp_stream_scatter(down.as<float>(), sorted + g.first_row, static_cast<const float*>(weights->data),
                               static_cast<float*>(out->data), g.rows, stream);
            CUDA_CHECK(cudaEventRecord(w.done, stream)); w.done_recorded = true;
            timing.done(i, stream);
            if (i + 2 < plan.groups.size()) enqueue(s, plan.groups[i + 2], w);
        }
        timing.end(stream);
        trace.save(metrics.payload_bytes-payload_before,metrics.cache_reused_bytes-reused_before);
        ++metrics.layers; metrics.groups += plan.groups.size();
        metrics.grouped_products += 2 * plan.groups.size();
        metrics.weight_window_bytes = 2 * window_bytes+prefixes[0].device.bytes+prefixes[1].device.bytes;
        metrics.activation_scratch_bytes = xq.bytes + gu.bytes + h.bytes + hq.bytes + down.bytes + maps.bytes;
    }
    PpStreamMetrics snapshot() {
        timing.collect(metrics);
        auto result=metrics;
        for(auto* memory:{&windows[0].device,&windows[1].device,&maps,&xq,&gu,&h,&hq,&down}) {
            result.device_alloc_ns+=memory->alloc_ns;result.device_alloc_calls+=memory->alloc_calls;
        }
        for(const auto& prefix:prefixes){result.device_alloc_ns+=prefix.device.alloc_ns;result.device_alloc_calls+=prefix.device.alloc_calls;}
        return result;
    }
};
std::mutex registry_mutex;
std::map<ggml_backend_cuda_context*, std::unique_ptr<Pipeline>> registry;
}
void pp_stream_run(ggml_backend_cuda_context& ctx, ggml_tensor* out) {
    Pipeline* p;
    {
        std::lock_guard<std::mutex> guard(registry_mutex);
        auto& entry = registry[&ctx]; if (!entry) entry = std::make_unique<Pipeline>(ctx.device);
        p = entry.get();
    }
    p->run(ctx, out);
}
void pp_prefetch_node(ggml_backend_cuda_context& ctx,const ggml_tensor* node) {
    int layer=0;const auto* view=pp_prefetch_view(node,layer);if(!view)return;
    std::lock_guard<std::mutex> guard(registry_mutex);
    const auto found=registry.find(&ctx);if(found==registry.end()||!found->second->packed_query)return;
    auto& pipeline=*found->second;
    pipeline.prefixes[size_t(layer)%2].load(view,pipeline.packed_query,pipeline.packed_owner,pipeline.copy,pipeline.metrics);
}
void pp_stream_release(ggml_backend_cuda_context* ctx) {
    std::lock_guard<std::mutex> guard(registry_mutex);
    auto i = registry.find(ctx);
    if (i != registry.end()) { CUDA_CHECK(cudaSetDevice(ctx->device)); CUDA_CHECK(cudaDeviceSynchronize()); registry.erase(i); }
}
PpStreamMetrics pp_stream_metrics(ggml_backend_cuda_context* ctx) {
    std::lock_guard<std::mutex> guard(registry_mutex);
    const auto i = registry.find(ctx); return i == registry.end() ? PpStreamMetrics{} : i->second->snapshot();
}
PpStreamMetrics pp_stream_backend_metrics(ggml_backend_t backend) {
    return pp_stream_metrics(static_cast<ggml_backend_cuda_context*>(backend->context));
}
bool pp_stream_get_metrics(ggml_backend_t backend, PpStreamMetrics* out) {
    if (!backend || !out) return false;
    *out = pp_stream_backend_metrics(backend);
    return true;
}
bool pp_stream_bind_transport(ggml_backend_t backend, PpTransportQuery query, void* owner) {
    if (!backend || !ggml_backend_is_cuda(backend) || (query && !owner)) return false;
    auto* ctx = static_cast<ggml_backend_cuda_context*>(backend->context);
    std::lock_guard<std::mutex> guard(registry_mutex);
    auto i = registry.find(ctx);
    if (i == registry.end() && !query) return true;
    CUDA_CHECK(cudaSetDevice(ctx->device));
    auto& entry = registry[ctx];
    if (!entry) entry = std::make_unique<Pipeline>(ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));
    CUDA_CHECK(cudaStreamSynchronize(entry->copy));
    entry->source_query = query; entry->source_owner = owner;
    return true;
}
bool pp_stream_enter_tg(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    auto* ctx = static_cast<ggml_backend_cuda_context*>(backend->context);
    std::lock_guard<std::mutex> guard(registry_mutex);
    const auto i = registry.find(ctx);
    if (i != registry.end()) {
        CUDA_CHECK(cudaSetDevice(ctx->device));
        i->second->enter_tg(ctx->stream());
    }
    return true;
}
bool pp_bind_packed(ggml_backend_t backend,PackedExpertQuery query,void* owner) {
    if(!backend||!ggml_backend_is_cuda(backend)||(query&&!owner))return false;
    auto* ctx=static_cast<ggml_backend_cuda_context*>(backend->context);
    std::lock_guard<std::mutex> guard(registry_mutex);
    if(!query&&registry.find(ctx)==registry.end())return true;
    CUDA_CHECK(cudaSetDevice(ctx->device));auto& entry=registry[ctx];
    if(!entry)entry=std::make_unique<Pipeline>(ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));CUDA_CHECK(cudaStreamSynchronize(entry->copy));
    entry->packed_query=query;entry->packed_owner=owner;return true;
}
}
