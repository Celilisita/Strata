#include "checkpoint.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {

uint64_t checksum(const std::vector<uint8_t>& bytes) {
    uint64_t value = 14695981039346656037ull;
    for (uint8_t byte : bytes) { value ^= byte; value *= 1099511628211ull; }
    return value;
}

std::size_t scalar_bytes(int type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_I32: return 4;
        case GGML_TYPE_F16: case GGML_TYPE_BF16: return 2;
        default: throw std::runtime_error("checkpoint dtype is not a supported scalar activation");
    }
}

std::string filename(std::size_t index) {
    std::ostringstream name;
    name << "checkpoint-" << std::setfill('0') << std::setw(3) << index << ".bin";
    return name.str();
}

std::vector<Checkpoint> manifest(const std::filesystem::path& directory) {
    std::ifstream input(directory / "checkpoints.tsv");
    if (!input) throw std::runtime_error("cannot read checkpoints.tsv");
    std::string line;
    std::getline(input, line);
    if (line != "strata-deepseek-v4-checkpoints-v1") throw std::runtime_error("invalid checkpoint manifest version");
    std::vector<Checkpoint> result;
    uint64_t total = 0;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        Checkpoint value;
        std::size_t index = 0;
        row >> index >> value.name >> value.dtype >> value.type;
        for (auto& dim : value.dimensions) row >> dim;
        for (auto& stride : value.source_strides) row >> stride;
        row >> value.bytes >> std::hex >> value.checksum >> std::dec >> value.file;
        std::string extra;
        if (!row || row >> extra || index != result.size() || value.file != filename(index) ||
            value.bytes > checkpoint_tensor_cap || result.size() >= 48) {
            throw std::runtime_error("malformed or oversized checkpoint manifest");
        }
        uint64_t logical_bytes = scalar_bytes(value.type);
        for (int64_t dim : value.dimensions) {
            if (dim < 1 || logical_bytes > checkpoint_tensor_cap / (uint64_t) dim)
                throw std::runtime_error("invalid checkpoint dimensions");
            logical_bytes *= (uint64_t) dim;
        }
        if (logical_bytes != value.bytes || total > checkpoint_total_cap - value.bytes)
            throw std::runtime_error("checkpoint logical size mismatch");
        total += value.bytes;
        result.push_back(std::move(value));
    }
    if (result.empty()) throw std::runtime_error("checkpoint manifest is empty");
    return result;
}

std::vector<uint8_t> read_binary(const std::filesystem::path& directory, const Checkpoint& value) {
    if (std::filesystem::file_size(directory / value.file) != value.bytes)
        throw std::runtime_error("checkpoint file size mismatch");
    std::ifstream input(directory / value.file, std::ios::binary);
    std::vector<uint8_t> bytes((std::size_t) value.bytes);
    input.read(reinterpret_cast<char*>(bytes.data()), (std::streamsize) bytes.size());
    if (!input || checksum(bytes) != value.checksum) throw std::runtime_error("checkpoint checksum mismatch");
    return bytes;
}

double number(const uint8_t* data, int type) {
    if (type == GGML_TYPE_F32) { float value; std::memcpy(&value, data, 4); return value; }
    if (type == GGML_TYPE_I32) { int32_t value; std::memcpy(&value, data, 4); return value; }
    uint16_t value;
    std::memcpy(&value, data, 2);
    if (type == GGML_TYPE_F16) return ggml_fp16_to_fp32(value);
    uint32_t bits = (uint32_t) value << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

} // namespace

CheckpointCollector::CheckpointCollector(std::filesystem::path directory, bool fusion_seam) :
    directory_(std::move(directory)), fusion_seam_(fusion_seam) {
    std::filesystem::create_directories(directory_);
}

bool CheckpointCollector::selected(const ggml_tensor* tensor) const {
    if (paused_) return false;
    static const char* names[] = {
        "hc_init", "hc_mixes-0", "hc_pre-0", "hc_post-0", "hc_comb-0", "hc_attn_pre-0",
        "attn_norm-0", "qr-0", "qr_norm-0", "q_norm-0", "q-0", "kv_norm-0", "kv-0",
        "attn_raw-0", "attn_derope-0", "attn_wo_a-0", "attn_out-0", "hc_attn_post-0",
        "hc_ffn_pre-0", "ffn_norm-0", "ffn_moe_logits-0", "ffn_moe_probs-0",
        "ffn_moe_topk-0", "ffn_moe_weights-0", "ffn_moe_weights_norm-0",
        "ffn_moe_weights_scaled-0", "ffn_moe_gate-0", "ffn_moe_up-0",
        "ffn_moe_swiglu-0", "ffn_moe_swiglu_limited-0", "ffn_moe_down-0",
        "ffn_moe_weighted-0", "ffn_moe_out-0", "ffn_shexp-0", "ffn_out-0", "l_last-0"
    };
    const std::string name = ggml_get_name(tensor);
    // Observing intermediates would split Gate/Up and suppress canonical fusion.
    // This explicit seam compares their fused GLU, Down and downstream results.
    if (fusion_seam_ && (name == "ffn_moe_gate-0" || name == "ffn_moe_up-0")) return false;
    return std::any_of(std::begin(names), std::end(names), [&](const char* selected) { return name == selected; });
}

bool CheckpointCollector::observe(ggml_tensor* tensor) {
    if (complete_ || failed()) return false;
    if (checkpoints_.size() >= 48) throw std::runtime_error("checkpoint count exceeds 48 tensors");
    Checkpoint value;
    value.name = ggml_get_name(tensor);
    value.type = tensor->type;
    value.dtype = ggml_type_name(tensor->type);
    // Record evaluated output storage; this is not a claim about the scheduler's executing backend.
    const ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer) throw std::runtime_error("checkpoint output buffer is unavailable");
    const char* buffer_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
    if (!buffer_name || std::strlen(buffer_name) > 128)
        throw std::runtime_error("checkpoint buffer type name is unavailable or exceeds 128 bytes");
    value.buffer_type = buffer_name;
    for (unsigned char character : value.buffer_type)
        if (character < 32 || character == 127) throw std::runtime_error("checkpoint buffer type name is not TSV-safe");
    value.buffer_is_host = ggml_backend_buffer_is_host(buffer);
    value.buffer_usage = (int) ggml_backend_buffer_get_usage(buffer);
    const std::size_t element = scalar_bytes(value.type);
    if (tensor->nb[0] != element) throw std::runtime_error("activation dimension zero is not contiguous");
    uint64_t bytes = element;
    for (int dim = 0; dim < 4; ++dim) {
        value.dimensions[dim] = tensor->ne[dim];
        value.source_strides[dim] = tensor->nb[dim];
        if (tensor->ne[dim] < 1 || bytes > checkpoint_tensor_cap / (uint64_t) tensor->ne[dim])
            throw std::runtime_error("selected activation exceeds the 64 MiB checkpoint cap");
        bytes *= (uint64_t) tensor->ne[dim];
    }
    if (total_bytes_ > checkpoint_total_cap - bytes) throw std::runtime_error("checkpoint total exceeds 128 MiB");
    value.bytes = bytes;
    std::vector<uint8_t> data((std::size_t) bytes);
    const std::size_t width = element * (std::size_t) tensor->ne[0];
    std::size_t destination = 0;
    // Compact logical rows, so padding and aliased view gaps never enter comparisons.
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            const std::size_t offset = (std::size_t) i3 * tensor->nb[3] + (std::size_t) i2 * tensor->nb[2];
            ggml_backend_tensor_get_2d(tensor, data.data() + destination, offset, width,
                                      (std::size_t) tensor->ne[1], tensor->nb[1], width);
            destination += width * (std::size_t) tensor->ne[1];
        }
    value.checksum = checksum(data);
    value.file = filename(checkpoints_.size());
    std::ofstream output(directory_ / value.file, std::ios::binary);
    output.write(reinterpret_cast<const char*>(data.data()), (std::streamsize) data.size());
    if (!output) throw std::runtime_error("cannot write checkpoint binary");
    total_bytes_ += bytes;
    complete_ = value.name == "l_last-0";
    checkpoints_.push_back(std::move(value));
    return !complete_;
}

bool CheckpointCollector::callback(ggml_tensor* tensor, bool ask, void* user) noexcept {
    auto& collector = *static_cast<CheckpointCollector*>(user);
    if (ask) return collector.complete_ || collector.failed() || collector.selected(tensor);
    try { return collector.observe(tensor); }
    catch (const std::exception& error) { collector.error_ = error.what(); }
    catch (...) { collector.error_ = "unknown checkpoint callback failure"; }
    return false;
}

void CheckpointCollector::write_manifest() const {
    std::ofstream output(directory_ / "checkpoints.tsv");
    output << "strata-deepseek-v4-checkpoints-v1\n";
    for (std::size_t i = 0; i < checkpoints_.size(); ++i) {
        const auto& value = checkpoints_[i];
        output << i << '\t' << value.name << '\t' << value.dtype << '\t' << value.type;
        for (auto dim : value.dimensions) output << '\t' << dim;
        for (auto stride : value.source_strides) output << '\t' << stride;
        output << '\t' << value.bytes << '\t' << std::hex << value.checksum << std::dec << '\t' << value.file << '\n';
    }
    if (!output) throw std::runtime_error("cannot write checkpoint manifest");
    std::ofstream backends(directory_ / "checkpoint-backends.tsv");
    backends << "strata-deepseek-v4-checkpoint-backends-v1\n"
             << "index\tname\tbuffer_type\tis_host\tbuffer_usage\n";
    for (std::size_t i = 0; i < checkpoints_.size(); ++i) {
        const auto& value = checkpoints_[i];
        backends << i << '\t' << value.name << '\t' << value.buffer_type << '\t'
                 << (value.buffer_is_host ? 1 : 0) << '\t' << value.buffer_usage << '\n';
    }
    if (!backends) throw std::runtime_error("cannot write checkpoint backend metadata");
}

CheckpointComparison compare_checkpoints(const std::filesystem::path& reference,
                                         const std::filesystem::path& candidate) {
    CheckpointComparison result;
    try {
        const auto left = manifest(reference), right = manifest(candidate);
        const std::size_t count = std::min(left.size(), right.size());
        for (std::size_t index = 0; index < count; ++index) {
            const auto& a = left[index]; const auto& b = right[index];
            if (a.name != b.name || a.type != b.type || a.dimensions != b.dimensions || a.bytes != b.bytes) {
                result.first_difference = a.name + ": metadata";
                return result;
            }
            const auto av = read_binary(reference, a), bv = read_binary(candidate, b);
            ++result.compared_tensors;
            const std::size_t width = scalar_bytes(a.type);
            for (std::size_t at = 0; at < av.size(); at += width) {
                const double x = number(av.data() + at, a.type), y = number(bv.data() + at, b.type);
                if (!std::isfinite(x) || !std::isfinite(y)) { result.has_nonfinite = true; continue; }
                const double difference = std::abs(x - y);
                result.max_absolute_error = std::max(result.max_absolute_error, difference);
                result.max_relative_error = std::max(result.max_relative_error,
                    difference / std::max(std::abs(x), 1e-12));
            }
            const auto mismatch = std::mismatch(av.begin(), av.end(), bv.begin());
            if (mismatch.first != av.end() || result.has_nonfinite) {
                result.first_difference = a.name + "#" + std::to_string(index);
                result.first_difference_byte = (uint64_t) std::distance(av.begin(), mismatch.first);
                return result;
            }
        }
        if (left.size() != right.size()) { result.first_difference = "checkpoint count"; return result; }
        if (left.back().name != "l_last-0") { result.first_difference = "layer-zero completion missing"; return result; }
        result.equal = true;
    } catch (const std::exception& error) { result.error = error.what(); }
    return result;
}

CheckpointData read_checkpoint_data(const std::filesystem::path& directory, const std::string& name) {
    const auto values = manifest(directory);
    if (std::count_if(values.begin(), values.end(), [&](const auto& v) { return v.name == name; }) != 1)
        throw std::runtime_error("matrix probe requires one checkpoint named " + name);
    const auto value = std::find_if(values.begin(), values.end(), [&](const auto& v) { return v.name == name; });
    return {*value, read_binary(directory, *value)};
}

CheckpointData read_checkpoint_occurrence(const std::filesystem::path& directory,
                                         const std::string& name, std::size_t occurrence) {
    if (occurrence > 1) throw std::runtime_error("mHC probe occurrence exceeds two layer subblocks");
    for (const auto& value : manifest(directory)) {
        if (value.name != name) continue;
        if (occurrence == 0) return {value, read_binary(directory, value)};
        --occurrence;
    }
    throw std::runtime_error("mHC probe checkpoint occurrence missing: " + name);
}

} // namespace strata::deepseek_v4
