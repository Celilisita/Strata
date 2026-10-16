#include "expert_source_pin.hpp"
#include "expert_source_pin_backend.hpp"

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

ExpertSourcePin::ExpertSourcePin(ggml_backend_t backend,
                               const std::vector<ggml_tensor*>& sources, uint64_t budget)
    {
    if (!backend || !budget || sources.empty() || sources.size() > 4096)
        throw std::runtime_error("source pin requires a backend and a bounded nonzero budget");
    const auto device = ggml_backend_get_device(backend);
    device_ = device;
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
    try {
        event_ = ggml_backend_event_new(device);
        if (!event_) throw std::runtime_error("source pin completion event allocation failed");
        uint64_t attempted_bytes = 0;
        // Select in canonical model order, independent of mmap addresses.
        for (const auto range : plan) {
            const uint64_t left = budget - attempted_bytes;
            const uint64_t bytes = std::min<uint64_t>(range.end - range.begin, left / page * page);
            if (!bytes) break;
            attempted_bytes += bytes;
            ++metrics_.attempts;
            if (register_readonly_source(reinterpret_cast<void*>(range.begin),
                                         static_cast<std::size_t>(bytes), metrics_.last_error)) {
                ranges_.push_back({range.begin, range.begin + static_cast<uintptr_t>(bytes)});
                metrics_.registered_bytes += bytes;
            } else {
                ++metrics_.failures;
            }
        }
    } catch (...) {
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

bool ExpertSourcePin::matches(ggml_backend_t backend, const std::vector<ggml_tensor*>& sources, uint64_t budget) const {
    if (!backend || ggml_backend_get_device(backend) != device_ || budget != metrics_.budget_bytes ||
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
