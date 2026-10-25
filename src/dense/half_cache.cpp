#include "strata/dense/half_cache.hpp"
#include "ggml-alloc.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace strata::dense {
namespace {

constexpr size_t host_staging_bytes = size_t(16) * 1024 * 1024;

bool selected(const TensorInfo &tensor) {
    return tensor.name.rfind("blk.", 0) == 0 && tensor.shape.size() == 2 &&
           (tensor.type == 8 || tensor.type == 12 || tensor.type == 14);
}

struct Candidate {
    const TensorInfo *source;
    size_t layer;
    int priority;
};

Candidate candidate(const TensorInfo &tensor, size_t n_layers) {
    const auto end = tensor.name.find('.', 4);
    if (end == std::string::npos)
        throw std::runtime_error("half cache missing block index: " + tensor.name);
    size_t layer = 0;
    const auto parsed = std::from_chars(tensor.name.data()+4, tensor.name.data()+end, layer);
    if (parsed.ec != std::errc() ||
        parsed.ptr != tensor.name.data()+end || layer >= n_layers)
        throw std::runtime_error("half cache invalid block index: " + tensor.name);
    const auto role = tensor.name.substr(end+1);
    const int priority = role == "ffn_gate.weight" || role == "ffn_up.weight" ? 0 :
                         role == "ffn_down.weight" ? 1 : 2;
    return {&tensor, layer, priority};
}

ggml_context *metadata(size_t count) {
    const size_t overhead = ggml_tensor_overhead();
    if (!overhead || count > size_t((std::numeric_limits<int>::max)()) ||
        count > (std::numeric_limits<size_t>::max)() / overhead - 32)
        throw std::runtime_error("half cache metadata size overflow");
    ggml_init_params params{overhead * (count + 32), nullptr, true};
    auto *context = ggml_init(params);
    if (!context) throw std::runtime_error("half cache metadata allocation failed");
    return context;
}

ggml_tensor *representation(ggml_context *context, const TensorInfo &source) {
    auto *cached = ggml_new_tensor_2d(context, GGML_TYPE_F16,
                                     int64_t(source.shape[0]), int64_t(source.shape[1]));
    ggml_set_name(cached, source.name.c_str());
    return cached;
}

size_t checked_product(uint64_t first, uint64_t second, const char *what) {
    if (!first || !second || first > (std::numeric_limits<size_t>::max)() / second)
        throw std::runtime_error(std::string("half cache extent overflow: ") + what);
    return size_t(first * second);
}

void upload(const GgufFile &file, const TensorInfo &source, ggml_tensor *destination) {
    const auto type = ggml_type(source.type);
    const auto *traits = ggml_get_type_traits(type);
    int block_elements = 0, block_bytes = 0;
    if (!traits || !traits->to_float || !traits->is_quantized ||
        !block_geometry(source.type, block_elements, block_bytes) ||
        traits->blck_size != block_elements || traits->type_size != size_t(block_bytes))
        throw std::runtime_error("half cache GGML quantization geometry differs: " + source.name);

    const size_t columns = size_t(source.shape[0]);
    const size_t rows = size_t(source.shape[1]);
    const size_t row_staging = checked_product(columns, sizeof(float) + sizeof(ggml_fp16_t),
                                              "host staging row");
    if (row_staging > host_staging_bytes)
        throw std::runtime_error("half cache row exceeds 16MiB host staging limit: " + source.name);
    const size_t chunk_rows = std::min(rows, host_staging_bytes / row_staging);
    const size_t chunk_elements = checked_product(columns, chunk_rows, "host staging chunk");
    std::vector<float> decoded(chunk_elements);
    std::vector<ggml_fp16_t> half(chunk_elements);
    const size_t quant_row_bytes = checked_product(columns / size_t(block_elements),
                                                   uint64_t(block_bytes), "quantized row");
    const size_t half_row_bytes = checked_product(columns, sizeof(ggml_fp16_t), "FP16 row");
    const auto *payload = file.tensor_data(source);
    for (size_t first = 0; first < rows; first += chunk_rows) {
        const size_t count = std::min(chunk_rows, rows - first);
        const size_t elements = columns * count;
        traits->to_float(payload + first * quant_row_bytes, decoded.data(), int64_t(elements));
        ggml_fp32_to_fp16_row(decoded.data(), half.data(), int64_t(elements));
        // The synchronous upload joins the copy before staging is overwritten.
        ggml_backend_tensor_set(destination, half.data(), first * half_row_bytes,
                                count * half_row_bytes);
    }
}
} // namespace

HalfCache::HalfCache(const GgufFile &file, const Spec &spec, ggml_backend_t backend,
                     size_t available_bytes, bool ffn_only) : budget_bytes_(available_bytes) {
    const auto begin = std::chrono::steady_clock::now();
    const auto finish = [&] {
        cold_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    };
    try {
        if (!backend) throw std::invalid_argument("half cache requires a backend");
        if (!spec.embedding || file.find(spec.embedding->name) != spec.embedding ||
            spec.layers.size() != spec.n_layers || spec.layers.empty())
            throw std::invalid_argument("half cache spec must borrow this complete GGUF inventory");
        const auto buffer_type = ggml_backend_get_default_buffer_type(backend);
        if (!buffer_type || ggml_backend_buft_is_host(buffer_type))
            throw std::invalid_argument("half cache requires a resident device buffer type");
        std::vector<Candidate> inventory;
        for (const auto &source : file.tensors()) {
            if (!selected(source)) continue;
            const auto item = candidate(source, spec.n_layers);
            if (ffn_only && item.priority > 1) continue;
            tensor_bytes(file, source);
            if (source.shape[0] > uint64_t((std::numeric_limits<int64_t>::max)()) ||
                source.shape[1] > uint64_t((std::numeric_limits<int64_t>::max)()))
                throw std::runtime_error("half cache dimensions exceed GGML limits: " + source.name);
            checked_product(checked_product(source.shape[0], source.shape[1], "FP16 elements"),
                            sizeof(ggml_fp16_t), "FP16 payload");
            inventory.push_back(item);
        }
        candidate_count_ = inventory.size();
        if (inventory.empty()) { finish(); return; }
        if (!available_bytes) { selection_status_ = "empty_budget"; finish(); return; }
        std::sort(inventory.begin(), inventory.end(), [](const auto &a, const auto &b) {
            return std::tie(a.priority, a.layer, a.source->name) <
                   std::tie(b.priority, b.layer, b.source->name);
        });
        // Probe metadata has no GPU buffer. Rejected candidates must not appear
        // in the final context: alloc_ctx would otherwise allocate them too.
        std::unique_ptr<ggml_context, decltype(&ggml_free)> probe(metadata(inventory.size()), ggml_free);
        std::vector<const TensorInfo *> sources;
        const size_t alignment = ggml_backend_buft_get_alignment(buffer_type);
        if (!alignment || (alignment & (alignment-1)))
            throw std::runtime_error("half cache invalid backend alignment");
        size_t planned_bytes = 0;
        for (const auto &item : inventory) {
            auto *cached = representation(probe.get(), *item.source);
            const size_t allocated = ggml_backend_buft_get_alloc_size(buffer_type, cached);
            if (allocated > (std::numeric_limits<size_t>::max)() - (alignment-1))
                throw std::runtime_error("half cache aligned allocation size overflow");
            const size_t padded = (allocated+alignment-1) & ~(alignment-1);
            // CUDA's allocator adds each independently aligned tensor size.
            // This guard also prevents overflow inside GGML's planning sum.
            if (padded > available_bytes-planned_bytes) continue;
            planned_bytes += padded;
            sources.push_back(item.source);
        }
        if (sources.empty()) { selection_status_ = "empty_budget"; finish(); return; }
        context_ = metadata(sources.size());
        tensors_.reserve(sources.size());
        for (const auto *source : sources) tensors_.emplace(source, representation(context_, *source));
        const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(context_, buffer_type);
        if (!required || required != planned_bytes || required > available_bytes)
            throw std::runtime_error("half cache final dry allocation differs from budgeted subset");
        probe.reset();
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(context_, buffer_type);
        if (!buffer_) {
            release();
            selection_status_ = "allocation_failed";
            finish();
            return;
        }
        bytes_ = ggml_backend_buffer_get_size(buffer_);
        if (bytes_ > available_bytes || bytes_ > required)
            throw std::runtime_error("half cache backend allocation exceeded its planned size");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto *source : sources) upload(file, *source, tensors_.at(source));
        selection_status_ = sources.size() == inventory.size() ? "complete" : "partial";
        finish();
    } catch (...) {
        release();
        throw;
    }
}

ggml_tensor *HalfCache::tensor(const TensorInfo *source) const noexcept {
    const auto found = tensors_.find(source);
    return found == tensors_.end() ? nullptr : found->second;
}

void HalfCache::release() noexcept {
    tensors_.clear();
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (context_) ggml_free(context_);
    buffer_ = nullptr;
    context_ = nullptr;
    bytes_ = 0;
}

HalfCache::~HalfCache() { release(); }
} // namespace strata::dense
