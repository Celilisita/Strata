#include "expert_source_pin.hpp"
#include "expert_source_pin_backend.hpp"
#include "source_pin_transaction.hpp"
#include "source_pin_admission.hpp"
#include "source_pin_stream_probe.hpp"
#include <cstdio>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace strata::deepseek_v4 {

uint64_t expert_source_pin_budget() {
    const char* text = std::getenv("STRATA_DSV4_SOURCE_PIN_MIB");
    if (!text) return 0;
    uint64_t mib = 0;
    const char* end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, mib);
    if (parsed.ec != std::errc{} || parsed.ptr != end || mib > 131072)
        throw std::runtime_error("source pin budget must be an integer in0..131072 MiB");
    return mib << 20;
}
bool expert_source_private_mapping_requested() {
    const char* value = std::getenv("STRATA_DSV4_SOURCE_PRIVATE_MAPPING");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("private source mapping must be0 or1");
}

bool expert_source_alias_requested() {
    const char* value = std::getenv("STRATA_GLM53_COLD_GPU_SOURCE_ALIAS");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("source GPU alias must be0 or1");
}

ExpertSourcePin::ExpertSourcePin(ggml_backend_t backend,
                               const std::vector<ggml_tensor*>& sources, uint64_t budget)
    {
    if (!backend || !budget || sources.empty() || sources.size() > 4096)
        throw std::runtime_error("source pin requires a backend and a bounded nonzero budget");
    const auto device = ggml_backend_get_device(backend);
    device_ = device;
    aliases_enabled_ = expert_source_alias_requested();
#ifdef _WIN32
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const uintptr_t page = info.dwPageSize;
#else
    const long page_size = sysconf(_SC_PAGESIZE);
    const uintptr_t page = page_size > 0 ? static_cast<uintptr_t>(page_size) : 0;
#endif
    if (!page) throw std::runtime_error("source pin could not determine the host page size");
    std::vector<Range> plan;
    for (const auto* tensor : sources) {
        if (!tensor || !tensor->data || !tensor->buffer ||
            !ggml_backend_buffer_is_host(tensor->buffer) ||
            ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
            throw std::runtime_error("source pin accepts only allocated canonical host weights");
        const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor->data);
        const std::size_t bytes = ggml_nbytes(tensor);
        const uintptr_t maximum = (std::numeric_limits<uintptr_t>::max)();
        if (bytes > maximum - begin || page - 1 > maximum - begin)
            throw std::runtime_error("source pin range overflows the host address space");
        sources_.push_back({begin, begin + bytes});
        // Register only complete pages inside the tensor. Edge bytes use the
        // existing slabs; no neighboring tensor or file mapping is borrowed.
        const uintptr_t first = (begin + page - 1) / page * page;
        const uintptr_t last = (begin + bytes) / page * page;
        if (first < last) plan.push_back({first, last});
    }
    auto ordered = plan;
    std::sort(ordered.begin(), ordered.end(), [](Range a, Range b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < ordered.size(); ++i)
        if (ordered[i].begin < ordered[i - 1].end)
            throw std::runtime_error("source pin ranges overlap");
    ranges_.reserve(plan.size());
    metrics_.budget_bytes = budget;
    const auto start = std::chrono::steady_clock::now();
    SourcePinTransaction transaction;
    uint64_t committed_bytes=0;bool pending_range=false;
    SourceAliasBudget first_budget{},before_budget{},after_budget{};
    uint64_t budget_queries=0,minimum_budget_free=UINT64_MAX,next_page_cost=0;
    bool budget_stopped=false,budget_query_ok=true,budget_available_seen=false;
    const char* budget_stage="not_required";uint32_t budget_error=0;
    bool before_budget_known=false,after_budget_known=false;
    auto report_budget=[&]() {
        if(!before_budget.required&&!after_budget.required)return;
        std::fprintf(stderr,"strata_source_alias_budget soft=1 scope=current_process known=%d pre_known=%d post_known=%d queries=%llu first_budget=%llu first_usage=%llu pre_budget=%llu pre_usage=%llu post_budget=%llu post_usage=%llu min_available=%llu next_page_cost=%llu stopped=%d stage=%s error=%u node=%u luid_low=%u luid_high=%u\n",
            int(budget_query_ok),int(before_budget_known),int(after_budget_known),(unsigned long long)budget_queries,
            (unsigned long long)first_budget.budget,(unsigned long long)first_budget.usage,
            (unsigned long long)before_budget.budget,(unsigned long long)before_budget.usage,
            (unsigned long long)after_budget.budget,(unsigned long long)after_budget.usage,
            (unsigned long long)(budget_available_seen?minimum_budget_free:0),
            (unsigned long long)next_page_cost,int(budget_stopped),
            budget_stage,budget_error,
            before_budget.node,before_budget.luid_low,before_budget.luid_high);
        std::fflush(stderr);
    };
    auto report=[&](const char* state) {
        std::fprintf(stderr,"strata_source_pin_transaction state=%s requested=%llu committed=%llu owned_registered=%llu sync_checks=%llu sync_failures=%llu rollback_ranges=%llu rollback_bytes=%llu stop_reason=%s stop_error=%u final_sync_error=%u\n",
            state,(unsigned long long)budget,(unsigned long long)committed_bytes,(unsigned long long)metrics_.registered_bytes,
            (unsigned long long)transaction.sync_checks,(unsigned long long)transaction.sync_failures,
            (unsigned long long)transaction.rollback_ranges,(unsigned long long)transaction.rollback_bytes,
            transaction.stop_reason,transaction.stop_error,transaction.final_sync_error);
        transaction.report();
        std::fflush(stderr);
    };
    try {
        event_ = ggml_backend_event_new(device);
        if (!event_) throw std::runtime_error("source pin completion event allocation failed");
        // CUDA backend event_new selects the backend device on this same thread.
        const int registration_device=source_alias_current_device();
        if(registration_device<0)throw std::runtime_error("source pin selected device unavailable");
        auto sync=[&](SourcePinApiResult& result) {
            // A commit/rollback covers the selected device AND both borrowed
            // backend queues, not a device-only fence followed by an unowned probe.
            if(aliases_enabled_)return fence_source_pin_streams(backend,"transaction_fence",metrics_.attempts,result);
            return synchronize_source_registration(registration_device,result);
        };
        auto clear_oom=[&](uint32_t expected,uint32_t& observed) {return clear_source_registration_error(registration_device,expected,observed);};
        transaction.begin(sync); // Never recover or clear a failed earlier computation.
        const auto admission=source_mapping_admission(sources,page,budget,registration_device,aliases_enabled_);
        admission.report(budget,0,"validated");
        auto query_budget=[&](SourceAliasBudget& info) {
            budget_query_ok=query_source_alias_budget(registration_device,info);
            budget_stage=info.stage;budget_error=info.error;
            if(&info==&before_budget)before_budget_known=budget_query_ok&&info.required;
            else after_budget_known=budget_query_ok&&info.required;
            if(info.required) {
                ++budget_queries;
                if(budget_queries==1)first_budget=info;
                if(budget_query_ok) {
                    const uint64_t available=info.budget>info.usage?info.budget-info.usage:0;
                    minimum_budget_free=budget_available_seen?std::min(minimum_budget_free,available):available;
                    budget_available_seen=true;
                }
            }
            if(!budget_query_ok)throw std::runtime_error(std::string("source alias budget unknown at ")+
                info.stage+": "+std::to_string(info.error));
        };
        if(aliases_enabled_)alias_device_=registration_device;
        auto rollback=[&](SourcePinApiResult& result) -> uint64_t {
            if(ranges_.empty())return 0;
            const auto range=ranges_.back();
            if(!unregister_source_checked(registration_device,reinterpret_cast<void*>(range.begin),result))return 0;
            // Primary success has already released ownership, even with unknown pending error.
            const uint64_t bytes=range.end-range.begin;
            ranges_.pop_back();metrics_.registered_bytes-=bytes;
            if(pending_range)pending_range=false;else committed_bytes-=bytes;
            if(range.alias)metrics_.alias_bytes-=bytes;
            return bytes;
        };
        uint64_t attempted_bytes = 0;
        // Select in canonical model order, independent of mmap addresses.
        for (const auto range : plan) {
            const uint64_t left = admission.limit - attempted_bytes;
            const uint64_t bytes = std::min<uint64_t>(range.end - range.begin, left / page * page);
            if (!bytes) break;
            if(aliases_enabled_) {
                query_budget(before_budget);
                if(before_budget.required) {
                    // Plan already contains complete OS pages; no guessed reserve or multiplier.
                    if(bytes%page)throw std::runtime_error("source alias page cost is not aligned");
                    next_page_cost=bytes;
                    const uint64_t available=before_budget.budget>before_budget.usage?
                        before_budget.budget-before_budget.usage:0;
                    if(bytes>available/page*page) {
                        budget_stopped=true;transaction.stop_reason="own_nonlocal_soft_budget";
                        break; // Keep the verified prefix; unregistered sources retain the original fallback.
                    }
                }
            }
            attempted_bytes += bytes;
            ++metrics_.attempts;
            SourcePinApiResult registration{},alias_result{};
            const bool registered=register_source_checked(registration_device,reinterpret_cast<void*>(range.begin),
                                         static_cast<std::size_t>(bytes),aliases_enabled_,registration);
            if(registered) {
                ranges_.push_back({range.begin,range.begin+static_cast<uintptr_t>(bytes),0});
                metrics_.registered_bytes+=bytes;pending_range=true;
            } else {
                ++metrics_.failures;metrics_.last_error=registration.primary;
            }
            transaction.allocation("register",registration,clear_oom);
            if(registered&&aliases_enabled_) {
                const uintptr_t alias=source_alias_checked(registration_device,reinterpret_cast<void*>(range.begin),alias_result);
                ranges_.back().alias=alias;
                if(alias_result.primary)metrics_.alias_last_error=alias_result.primary;
                if(!alias)++metrics_.alias_failures;
                if(alias)metrics_.alias_bytes+=bytes;
                transaction.allocation("alias",alias_result,clear_oom);
            }
            const bool more=transaction.complete(registered,registration.primary,alias_result.primary,sync,rollback,clear_oom);
            committed_bytes=metrics_.registered_bytes;pending_range=false;
            if(registered&&more&&aliases_enabled_) {
                query_budget(after_budget);
            }
            if(!more) {
                if(!metrics_.last_error)metrics_.last_error=transaction.stop_error;
                break;
            }
        }
        admission.report(budget,metrics_.registered_bytes,"registered");
        report_budget();
        report(budget_stopped?"verified_budget_prefix":transaction.stop_error?"verified_oom_fallback":"committed");
    } catch (...) {
        report_budget();
        report("failed");
        release();
        throw;
    }
    std::sort(ranges_.begin(), ranges_.end(), [](Range a, Range b) { return a.begin < b.begin; });
    metrics_.registration_wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

ExpertSourcePin::~ExpertSourcePin() noexcept { release(); }

void ExpertSourcePin::release() noexcept {
    if (busy_ && event_) ggml_backend_event_synchronize(event_);
    busy_ = false;
    // DMA events do not cover mapped compute-stream reads. Drain their device
    // before this same registration owner can unregister the borrowed pages.
    if (aliases_enabled_ && alias_device_ >= 0 && metrics_.alias_bytes &&
        !drain_source_alias_readers(alias_device_, metrics_.alias_last_error))
        GGML_ABORT("source GPU alias readers did not retire: CUDA error %u", metrics_.alias_last_error);
    for (const auto range : ranges_) unregister_readonly_source(reinterpret_cast<void*>(range.begin));
    ranges_.clear();
    if (event_) ggml_backend_event_free(event_);
    event_ = nullptr;
}

ExpertSourcePin::Segment ExpertSourcePin::segment(const void* source, std::size_t bytes) const {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(source);
    if (bytes > (std::numeric_limits<uintptr_t>::max)() - begin)
        throw std::runtime_error("source pin upload range overflows");
    const auto next = std::upper_bound(ranges_.begin(), ranges_.end(), begin,
        [](uintptr_t address, Range range) { return address < range.begin; });
    if (next != ranges_.begin()) {
        const auto range = *(next - 1);
        if (begin < range.end) return {std::min<std::size_t>(bytes, range.end - begin), true};
    }
    return {next == ranges_.end() ? bytes : std::min<std::size_t>(bytes, next->begin - begin), false};
}

uintptr_t ExpertSourcePin::borrow_device_alias(const void* source, std::size_t bytes) const {
    if (!aliases_enabled_ || !bytes) return 0;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(source);
    const uintptr_t maximum = (std::numeric_limits<uintptr_t>::max)();
    if (bytes > maximum - begin) return 0;
    const auto next = std::upper_bound(ranges_.begin(), ranges_.end(), begin,
        [](uintptr_t address, Range range) { return address < range.begin; });
    if (next == ranges_.begin()) return 0;
    const auto& range = *(next - 1);
    if (!range.alias || begin < range.begin || begin >= range.end || bytes > range.end - begin) return 0;
    const uintptr_t offset = begin - range.begin;
    if (offset > maximum - range.alias || bytes > maximum - (range.alias + offset)) return 0;
    return range.alias + offset;
}

bool ExpertSourcePin::matches(ggml_backend_t backend, const std::vector<ggml_tensor*>& sources, uint64_t budget) const {
    if (!backend || ggml_backend_get_device(backend) != device_ || budget != metrics_.budget_bytes || aliases_enabled_ != expert_source_alias_requested() ||
        sources.size() != sources_.size()) return false;
    for (size_t i = 0; i < sources.size(); ++i) {
        if (!sources[i] || reinterpret_cast<uintptr_t>(sources[i]->data) != sources_[i].begin ||
            ggml_nbytes(sources[i]) != sources_[i].end - sources_[i].begin) return false;
    }
    return true;
}
void ExpertSourcePin::submitted(ggml_backend_t backend) {
    if (!backend || ggml_backend_get_device(backend) != device_)
        throw std::runtime_error("source registration completion device differs");
    ggml_backend_event_record(event_, backend);
    busy_ = true;
}

} // namespace strata::deepseek_v4
