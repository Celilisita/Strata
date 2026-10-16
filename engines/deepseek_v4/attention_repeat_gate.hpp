#pragma once
#include "attention_repeat.hpp"
#include "long_prefix_compare.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
namespace strata::deepseek_v4 {
class AttentionRepeatReceipt {
public:
    explicit AttentionRepeatReceipt(Runtime& runtime) {
        auto reg = ggml_backend_dev_backend_reg(runtime.device());
        proc_ = reinterpret_cast<AttentionRepeatProc>(ggml_backend_reg_get_proc_address(reg, "strata_attention_repeat"));
        if (!proc_) throw std::runtime_error("attention repeat unavailable");
        command(AttentionRepeatAction::Initialize);
    }
    // The owning runtime/context is declared after this receipt in the driver.
    ~AttentionRepeatReceipt() { if (packet_.handle) { packet_.action = AttentionRepeatAction::Release; proc_(&packet_); } }
    void begin() { command(AttentionRepeatAction::Begin); }
    std::vector<char> finish(Runtime& runtime, const std::filesystem::path& path) {
        llama_synchronize(runtime.context()); command(AttentionRepeatAction::End);
        std::vector<char> data(attention_repeat_bytes);
        packet_.bytes = data.data(); packet_.capacity = data.size(); command(AttentionRepeatAction::Read);
        packet_.bytes = nullptr;
        std::ofstream saved(path, std::ios::binary); saved.write(data.data(), data.size());
        if (!saved) throw std::runtime_error("attention receipt write failed");
        return data;
    }
    size_t k_bytes() const { return packet_.k_bytes; }
    size_t mask_bytes() const { return packet_.mask_bytes; }
private:
    void command(AttentionRepeatAction action) {
        packet_.action = action;
        if (!proc_(&packet_)) throw std::runtime_error("attention receipt error " + std::to_string(packet_.error));
    }
    AttentionRepeatProc proc_ = nullptr;
    AttentionRepeatPacket packet_{AttentionRepeatAction::Initialize};
};
inline uint64_t attention_byte_differences(const char* a, const char* b, size_t bytes) {
    uint64_t result = 0; for (size_t k = 0; k < bytes; ++k) result += a[k] != b[k]; return result;
}
inline uint64_t attention_repeat_report(const std::vector<char>& bytes, std::ostream& report) {
    uint64_t failed = 0;
    report << "\"repeat_samples\":[";
    const auto* original = bytes.data()+attention_repeat_output_offset;
    for (int sample = 1; sample <= attention_repeat_count; ++sample) {
        const auto* repeated = original+sample*attention_repeat_row_bytes;
        uint64_t different = 0; double maximum = 0;
        for (size_t n = 0; n < attention_repeat_values; ++n) {
            float a = 0, b = 0; std::memcpy(&a, original+n*4, 4); std::memcpy(&b, repeated+n*4, 4);
            if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("nonfinite attention repeat");
            if (std::memcmp(original+n*4, repeated+n*4, 4)) { ++different; maximum = std::max(maximum, std::abs(double(a)-b)); }
        }
        if (sample > 1) report << ',';
        report << "{\"sample\":" << sample << ",\"differing_values\":" << different << ",\"max_abs\":" << maximum << '}';
        failed += different != 0;
    }
    report << ']'; return failed;
}
inline int prefix_attention_repeat_cases(Runtime& runtime, std::unique_ptr<AttentionRepeatReceipt>& receipt,
        const Options& options, PrefixReference& reference, const std::filesystem::path& output,
        std::ostream& report, std::string& phase) {
    receipt = std::make_unique<AttentionRepeatReceipt>(runtime);
    std::vector<char> first;
    size_t k_bytes = 0, mask_bytes = 0;
    uint64_t failed = 0;
    std::string error;
    for (int mode = 0; mode < 2; ++mode) {
        if (mode) {
            session_set_source_pin(131072); session_set_device_cache(true); session_set_frequency(true);
            if (!runtime.reconfigure_context(options, error)) throw std::runtime_error(error);
        }
        phase = mode ? "device_frequency_fa37" : "bounded_staging_fa37";
        receipt->begin();
        const auto difference = prefix_case(runtime, reference, output, phase, false, 1, mode == 0);
        auto data = receipt->finish(runtime, output / (phase+"-attention.bin"));
        if (mode) report << ',';
        report << "{\"name\":" << session_json(phase) << ",\"logits_difference\":";
        prefix_write_difference(report, difference);
        report << ",\"k_bytes\":" << receipt->k_bytes() << ",\"mask_bytes\":" << receipt->mask_bytes() << ',';
        failed += attention_repeat_report(data, report);
        if (!mode) { first = std::move(data); k_bytes = receipt->k_bytes(); mask_bytes = receipt->mask_bytes(); }
        else {
            if (k_bytes != receipt->k_bytes() || mask_bytes != receipt->mask_bytes()) throw std::runtime_error("attention input shape changed");
            uint64_t visible_k_differences = 0, visible_rows = 0;
            for (size_t row = 0; row < mask_bytes/2; ++row) {
                uint16_t a = 0, b = 0;
                std::memcpy(&a, first.data()+attention_repeat_mask_offset+row*2, 2);
                std::memcpy(&b, data.data()+attention_repeat_mask_offset+row*2, 2);
                if ((a & 0x7c00) != 0x7c00 || (b & 0x7c00) != 0x7c00) {
                    ++visible_rows;
                    visible_k_differences += attention_byte_differences(first.data()+attention_repeat_k_offset+row*1024,
                        data.data()+attention_repeat_k_offset+row*1024, 1024);
                }
            }
            report << ",\"input_difference_bytes\":{\"q\":" << attention_byte_differences(first.data(), data.data(), attention_repeat_row_bytes)
                << ",\"mask\":" << attention_byte_differences(first.data()+attention_repeat_mask_offset, data.data()+attention_repeat_mask_offset, mask_bytes)
                << ",\"sinks\":" << attention_byte_differences(first.data()+attention_repeat_sink_offset, data.data()+attention_repeat_sink_offset, 256)
                << ",\"visible_k\":" << visible_k_differences << "},\"visible_k_rows\":" << visible_rows
                << ",\"original_output_difference_bytes\":" << attention_byte_differences(first.data()+attention_repeat_output_offset,
                    data.data()+attention_repeat_output_offset, attention_repeat_row_bytes);
        }
        report << '}'; report.flush();
    }
    report << "],\"same_input_repeat_failures\":" << failed << ",\"diagnostic_only\":true,\"performance_target_met\":false}";
    return failed ? 2 : 0;
}
}
