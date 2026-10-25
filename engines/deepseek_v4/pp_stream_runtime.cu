#include "pp_stream_cuda.cuh"
#include "source_pin_stream_probe.hpp"
#include <cuda.h>
#include "pp_stream_plan.hpp"
#include "pp_miss_coalesce_runtime.cuh"
#include "pp_miss_coalesce_upload.cuh"
#include "../glm5_next/pp_transport_verify.cuh"
#include "pp_stream_transfer.cuh"
#include "pp_stream_timing.cuh"
#include "../glm5_next/pp_weights_view.cuh"
#include "../glm5_next/pp_source_window_planar.cuh"
#include "pp_packed_upload.cuh"
#include "../glm5_next/pp_prefix.cuh"
#include "pp_host_trace.hpp"
#include "pp_host_tile_counts.hpp"
#include "pp_plan_receipt_capture.hpp"
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
    bool packed = false;size_t packed_stride=0;
    bool owned_planar=false;
    const void* borrowed=nullptr;std::array<const void*,3> planar{};
    cudaEvent_t dependency=nullptr;
    const void* payload()const{return borrowed?borrowed:device.p;}
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
    PpPlanReceiptStore plan_receipt;
    PpHostTrace trace;
    PpStreamTiming timing;
    PpStreamEdges edges;
    PpTransportQuery source_query = nullptr;
    void* source_owner = nullptr;
    PackedExpertQuery packed_query = nullptr;
    void* packed_owner = nullptr;
    std::array<PpPackedPrefix<Memory>,2> prefixes;
    glm5_next::PpPrefix<Memory> glm_prefix;
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
        plan_receipt.seal();
        for (auto& window : windows) window.device.clear();
        for(auto& prefix:prefixes)prefix.clear();
        glm_prefix.collect(metrics);glm_prefix.clear();
        maps.clear(); xq.clear(); gu.clear(); h.clear(); hq.clear(); down.clear();
    }
    void enqueue(const PpStreamSources& s, const PpStreamGroup& g, Window& w,bool tight_fences,bool source_planar,bool verify_transport) {
        w.packed=false;w.packed_stride=0;w.borrowed=nullptr;w.planar={};w.owned_planar=false;w.dependency=w.ready;
        if(g.cache_first_slot>=0) {
            // Ready cache rows are consumed directly by canonical MMQ. Hits
            // delete host uploads and the intermediate device weight copy.
            metrics.cache_reused_bytes+=uint64_t(g.experts)*(2*s.gate->nb[2]+s.down->nb[2]);
            if(tight_fences)w.dependency=nullptr;else {CUDA_CHECK(cudaEventRecord(w.ready,copy));w.ready_recorded=true;}
            return;
        }
        if(g.source_keys.empty())if(const void* prefix=glm_prefix.borrow(s,g,metrics)) {
            if(glm_prefix.planar())for(int role=0;role<3;++role)w.planar[role]=glm_prefix.role_payload(role,g.first_expert);
            else {w.borrowed=prefix;w.packed=true;w.packed_stride=glm_prefix.payload_stride();}
            if(tight_fences)w.dependency=glm_prefix.ready_for_group(s,g,metrics);else {CUDA_CHECK(cudaEventRecord(w.ready,copy));w.ready_recorded=true;}return;
        }
        const size_t half = s.gate->nb[2], gu_stride = 2 * half, d_stride = s.down->nb[2];
        const auto* gate = static_cast<const char*>(s.gate->data) + g.first_expert * half;
        const auto* up = static_cast<const char*>(s.up->data) + g.first_expert * half;
        const auto* dn = static_cast<const char*>(s.down->data) + g.first_expert * d_stride;
        // Waits are issued only after the previous reader has been recorded.
        // Waiting on an unrecorded CUDA event would provide no protection.
        if (w.done_recorded) CUDA_CHECK(cudaStreamWaitEvent(copy, w.done, 0));
        if(pp_upload_coalesced_misses(s,g,w,copy,source_query,source_owner,edges,metrics,pinned_span)) {
            if(verify_transport)glm5_next::pp_transport_verify_owned(s,g,w,copy,metrics);return;
        }
        if(packed_query) {
            const PpPrefetchedSpan* prefetched=nullptr;
            for(const auto& prefix:prefixes)if(prefix.span.gate==s.gate)prefetched=&prefix.span;
            pp_packed_upload(s,g,packed_query(s.gate,g.first_expert,g.experts,packed_owner),w.device.p,copy,metrics,prefetched);
            w.packed=true;w.packed_stride=2*s.gate->nb[2]+s.down->nb[2];CUDA_CHECK(cudaEventRecord(w.ready,copy));w.ready_recorded=true;return;
        }
        const size_t down_offset = size_t(s.window_experts) * gu_stride;
        if(source_planar)glm5_next::pp_bind_source_window_planar(s,g,w);
        const uint64_t source_payload_before=metrics.payload_bytes;
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
        matrix(gate, half, w.owned_planar?half:gu_stride, 0, 0);
        matrix(up, half, w.owned_planar?half:gu_stride, w.owned_planar?size_t(s.window_experts)*half:half, 1);
        matrix(dn, d_stride, d_stride, down_offset, 2);
        if(w.owned_planar){++metrics.source_planar_groups;metrics.source_planar_bytes+=metrics.payload_bytes-source_payload_before;}
        CUDA_CHECK(cudaEventRecord(w.ready, copy)); w.ready_recorded = true;
        if(verify_transport)glm5_next::pp_transport_verify_owned(s,g,w,copy,metrics);
    }
    void run(ggml_backend_cuda_context& ctx, ggml_tensor* out) {
        const auto s = pp_stream_sources(out);
        const bool coalesce=s.gate->ne[2]==288&&pp_miss_coalesce_requested();metrics.miss_coalesce_active=coalesce;
        const bool verify_transport=glm5_next::pp_transport_verify_requested();
        if(coalesce&&plan_receipt.active())throw std::runtime_error("PP receipt-v1 cannot describe coalesced source keys");
        const bool source_planar=glm5_next::pp_source_window_planar_requested(s);metrics.source_planar_active=source_planar;
        const bool tight_fences=s.gate->ne[2]==288&&glm5_next::pp_prefix_true_fences();
        const bool reuse_host_counts=s.gate->ne[2]==288&&pp_reuse_host_tile_counts_requested();
        const auto payload_before=metrics.payload_bytes,reused_before=metrics.cache_reused_bytes;
        auto product=[&](auto&&... args){trace.product([&]{pp_stream_product(std::forward<decltype(args)>(args)...);});};
        for (const auto* bank : {s.gate, s.up, s.down})
            if (!bank || !bank->data || !bank->buffer || !ggml_backend_buffer_is_host(bank->buffer))
                throw std::runtime_error("PP execution requires live immutable host banks; reserve-only metadata cannot execute");
        if (s.cache) {
            const std::array<const ggml_tensor*,3> originals{s.gate,s.up,s.down};
            if (s.cache->source != originals || !s.cache->directory || !s.cache->directory->data ||
                s.cache->directory->type != GGML_TYPE_I32 || ggml_nelements(s.cache->directory) != s.gate->ne[2])
                throw std::runtime_error("PP cache source/key projection differs");
            for (int role=0;role<3;++role) {
                const auto* bank=s.cache->bank[role];
                if (!bank || !bank->data || bank->type != originals[role]->type ||
                    bank->nb[2] != originals[role]->nb[2]) throw std::runtime_error("PP cached payload layout differs");
            }
        }
        auto* x = out->src[0]; auto* ids = out->src[1]; auto* weights = out->src[2];
        if (!is_pp_stream(out) || !ggml_is_contiguous(x) || !ggml_is_contiguous(ids) ||
            !ggml_is_contiguous(weights) || !ggml_is_contiguous(out))
            throw std::runtime_error("PP streaming inputs must be contiguous");
        const int tokens = int(x->ne[1]), topk = int(ids->ne[0]), experts = int(s.gate->ne[2]), pairs = tokens * topk;
        const auto stream = ctx.stream();
        const char* precision = std::getenv("GGML_CUDA_MMQ_PREC");
        const bool fp4 = s.gate->type == GGML_TYPE_MXFP4 && blackwell_mma_available(ggml_cuda_info().devices[device].cc) &&
                         !(precision && std::strcmp(precision, "q8") == 0);
        route_host.reserve(pairs * sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(route_host.p, ids->data, pairs * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++metrics.route_host_reads;
        if(tight_fences)for(auto& window:windows)window.done_recorded=false;
        timing.collect(metrics);
        auto plan = make_pp_stream_plan(route_host.as<int32_t>(), tokens, topk, experts,
                                              s.window_experts, 2 * s.gate->nb[2] + s.down->nb[2],
                                              s.cache?static_cast<const int32_t*>(s.cache->directory->data):nullptr,glm_prefix.boundary(s));
        const auto regroup=coalesce?pp_prepare_miss_coalesce(s,plan,glm_prefix,packed_query!=nullptr,source_planar):PpMissCoalesceResult{};
        int last_prefix_group=-1;
        if(tight_fences)for(size_t i=0;i<plan.groups.size();++i)if(plan.groups[i].source_keys.empty()&&plan.groups[i].cache_first_slot<0&&
            plan.groups[i].first_expert+plan.groups[i].experts<=glm_prefix.boundary(s))last_prefix_group=int(i);
        if(s.cache&&s.cache->prefill_counts&&s.cache->prefill_tokens){
            if(experts!=288)std::fill_n(s.cache->prefill_counts,experts,0u);
            for(int pair=0;pair<pairs;++pair)++s.cache->prefill_counts[route_host.as<int32_t>()[pair]];
            if(experts==288)*s.cache->prefill_tokens+=tokens;else *s.cache->prefill_tokens=tokens;
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
            s.gate->type, true, 4096, 4096, 4096, tokens, pairs, topk, stream);
        else quantize_scatter_mmq_q8_1_cuda(static_cast<const float*>(x->data), inverse, xq.p,
            s.gate->type, 4096, 4096, 4096, tokens, pairs, topk, stream);
        CUDA_CHECK(cudaMemsetAsync(out->data, 0, ggml_nbytes(out), stream));
        for (size_t i = 0; i < std::min<size_t>(2, plan.groups.size()); ++i) enqueue(s, plan.groups[i], windows[i],tight_fences,source_planar,verify_transport);
        const char* ragged_setting=std::getenv("STRATA_DSV4_PP_RAGGED_MMQ");
        const bool ragged=!ragged_setting||std::strcmp(ragged_setting,"0")!=0;
        uint32_t receipt_layer=PpPlanReceiptStore::no_layer;
        if(plan_receipt.active()) {
            uint64_t missing[3]{};glm_prefix.receipt_missing(s,missing);
            const int boundary=glm_prefix.boundary(s);
            const char* direct=std::getenv("STRATA_DSV4_PP_DIRECT_CACHE_GROUPS");
            const uint32_t states=uint32_t(s.cache!=nullptr)|(uint32_t(boundary>0)<<1)|
                (uint32_t(boundary>0&&glm_prefix.planar())<<2)|(uint32_t(packed_query!=nullptr)<<3)|
                (uint32_t(s.cache&&!(direct&&!std::strcmp(direct,"0")))<<4)|(uint32_t(ragged)<<5)|
                (uint32_t(tight_fences)<<6)|(uint32_t(source_planar)<<7);
            receipt_layer=plan_receipt.plan(s,plan,route_host.as<int32_t>(),boundary,missing,states);
        }
        for (size_t i = 0; i < plan.groups.size(); ++i) {
            const auto& g = plan.groups[i]; auto& w = windows[i % 2];
            timing.waiting(i, stream);
            if(w.dependency)CUDA_CHECK(cudaStreamWaitEvent(stream, w.dependency, 0));
            timing.ready(i, stream);
            const auto* bounds = maps.as<int32_t>() + bound_offsets[i];
            const auto* tile_prefix=ragged?maps.as<int32_t>()+ragged_offsets[i]:nullptr;
            const auto* host_bounds=ragged?g.bounds.data():nullptr;
            const auto* host_tile_counts=ragged&&reuse_host_counts?host+ragged_offsets[i]:nullptr;
            // MMQ activation blocks are K-major: [K-block][compact row].
            // A group advances one block per row and retains the full route
            // count as its K-block stride, rather than using a row-major offset.
            const auto bank_view=pp_weights(s,g,w);
            if(receipt_layer!=PpPlanReceiptStore::no_layer) {
                const auto kind=g.cache_first_slot>=0?PpPlanSourceKind::direct_cache:
                    w.owned_planar?PpPlanSourceKind::source_planar:w.planar[0]?PpPlanSourceKind::prefix_planar:
                    w.borrowed?PpPlanSourceKind::prefix_packed:w.packed?PpPlanSourceKind::packed_inventory:
                    PpPlanSourceKind::source_interleaved;
                plan_receipt.kind(receipt_layer,i,kind,bank_view.separate);
            }
            if(bank_view.separate) {
                product(ctx,bank_view.role[0],xq.as<char>(g.first_row*sizeof(block_q8_1_mmq)),bounds,identity,
                    gu.as<float>(),g.experts,g.rows,pairs,g.max_rows,4096,2048,bank_view.gate_stride,fp4,stream,4096,tile_prefix,host_bounds,s.gate->type,host_tile_counts);
                product(ctx,bank_view.role[1],xq.as<char>(g.first_row*sizeof(block_q8_1_mmq)),bounds,identity,
                    gu.as<float>()+2048,g.experts,g.rows,pairs,g.max_rows,4096,2048,bank_view.gate_stride,fp4,stream,4096,tile_prefix,host_bounds,s.up->type,host_tile_counts);
                ++metrics.grouped_products;
                if(w.owned_planar)++metrics.source_planar_extra_GU_products;
            }else product(ctx,bank_view.role[0],xq.as<char>(g.first_row*sizeof(block_q8_1_mmq)),bounds,identity,
                    gu.as<float>(),g.experts,g.rows,pairs,g.max_rows,4096,4096,bank_view.gate_stride,fp4,stream,0,tile_prefix,host_bounds,s.gate->type,host_tile_counts);
            pp_stream_swiglu(gu.as<float>(), h.as<float>(), g.rows, s.clamp, stream);
            CUDA_CHECK(cudaMemsetAsync(hq.as<char>(quant_bytes(g.rows, 2048, fp4) - 128 * sizeof(block_q8_1_mmq)),
                                       0, 128 * sizeof(block_q8_1_mmq), stream));
            if (fp4) quantize_mmq_fp4_cuda(h.as<float>(), nullptr, hq.p, nullptr, s.down->type,
                true, 2048, 2048, int64_t(g.rows) * 2048, int64_t(g.rows) * 2048, 2048, g.rows, 1, 1, stream);
            else quantize_mmq_q8_1_cuda(h.as<float>(), nullptr, hq.p, s.down->type,
                2048, 2048, int64_t(g.rows) * 2048, int64_t(g.rows) * 2048, 2048, g.rows, 1, 1, stream);
            product(ctx, bank_view.role[2], hq.p, bounds, identity,
                down.as<float>(), g.experts, g.rows, g.rows, g.max_rows, 2048, 4096,
                bank_view.down_stride, fp4, stream,0,tile_prefix,host_bounds,s.down->type,host_tile_counts);
            pp_stream_scatter(down.as<float>(), sorted + g.first_row, static_cast<const float*>(weights->data),
                               static_cast<float*>(out->data), g.rows, stream, topk);
            if(!tight_fences||(!glm5_next::pp_window_borrows_prefix(w)&&g.cache_first_slot<0)){CUDA_CHECK(cudaEventRecord(w.done,stream));w.done_recorded=true;}
            if(glm5_next::pp_window_borrows_prefix(w)&&(!tight_fences||int(i)==last_prefix_group))glm_prefix.consumed(stream);
            timing.done(i, stream);
            if (i + 2 < plan.groups.size()) enqueue(s, plan.groups[i + 2], w,tight_fences,source_planar,verify_transport);
        }
        if(tight_fences&&glm_prefix.boundary(s)>0&&(last_prefix_group<0||glm_prefix.segment_ready_active(s)))CUDA_CHECK(cudaStreamWaitEvent(stream,glm_prefix.ready_event(),0));
        plan_receipt.done(receipt_layer);
        glm_prefix.end_layer();
        timing.end(stream);
        trace.save(metrics.payload_bytes-payload_before,metrics.cache_reused_bytes-reused_before);
        ++metrics.layers; metrics.groups += plan.groups.size();
        if(regroup.changed()){++metrics.miss_coalesced_layers;metrics.miss_coalesced_groups_before+=regroup.groups_before;metrics.miss_coalesced_groups_after+=regroup.groups_after;}
        metrics.grouped_products += 2 * plan.groups.size();
        metrics.weight_window_bytes = windows[0].device.bytes+windows[1].device.bytes+prefixes[0].device.bytes+prefixes[1].device.bytes+glm_prefix.device.bytes;
        metrics.activation_scratch_bytes = xq.bytes + gu.bytes + h.bytes + hq.bytes + down.bytes + maps.bytes;
    }
    PpStreamMetrics snapshot() {
        timing.collect(metrics);glm_prefix.collect(metrics);
        auto result=metrics;
        for(auto* memory:{&windows[0].device,&windows[1].device,&maps,&xq,&gu,&h,&hq,&down}) {
            result.device_alloc_ns+=memory->alloc_ns;result.device_alloc_calls+=memory->alloc_calls;
        }
        for(const auto& prefix:prefixes){result.device_alloc_ns+=prefix.device.alloc_ns;result.device_alloc_calls+=prefix.device.alloc_calls;}
        result.device_alloc_ns+=glm_prefix.device.alloc_ns;result.device_alloc_calls+=glm_prefix.device.alloc_calls;
        return result;
    }
};
std::mutex registry_mutex;
std::map<ggml_backend_cuda_context*, std::unique_ptr<Pipeline>> registry;
}
#include "source_pin_stream_probe.inc"
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
    const auto found=registry.find(&ctx);if(found==registry.end())return;
    auto& pipeline=*found->second;
    if(view->source[0]&&view->source[0]->ne[2]==288) {
        pipeline.glm_prefix.load(view,pipeline.source_query,pipeline.source_owner,pipeline.copy,pipeline.metrics,pinned_span);return;
    }
    if(pipeline.packed_query)pipeline.prefixes[size_t(layer)%2].load(view,pipeline.packed_query,pipeline.packed_owner,pipeline.copy,pipeline.metrics);
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
bool pp_plan_receipt_begin(ggml_backend_t backend,uint64_t nonce,bool enabled) {
    if(!backend||!ggml_backend_is_cuda(backend))return false;
    auto* ctx=static_cast<ggml_backend_cuda_context*>(backend->context);
    std::lock_guard<std::mutex> guard(registry_mutex);const auto i=registry.find(ctx);
    return i==registry.end()?!enabled:i->second->plan_receipt.begin(nonce,enabled);
}
bool pp_plan_receipt_read(ggml_backend_t backend,const PpPlanReceiptPod** out) {
    if(!backend||!out||!ggml_backend_is_cuda(backend))return false;
    auto* ctx=static_cast<ggml_backend_cuda_context*>(backend->context);
    std::lock_guard<std::mutex> guard(registry_mutex);const auto i=registry.find(ctx);
    *out=i==registry.end()?nullptr:i->second->plan_receipt.read();return true;
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
