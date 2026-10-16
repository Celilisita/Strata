#include "checkpoint.hpp"
#include "mapped_cold_worker.hpp"
#include "ggml-alloc.h"
#include "llama.h"
#include "llama-model.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace strata::deepseek_v4;
using Clock = std::chrono::steady_clock;
constexpr std::size_t row_bytes = 4096 * sizeof(float);
using Down = std::array<float, 6 * 4096>;

struct Inputs {
    std::array<float, 4096> norm{};
    std::array<int32_t, 6> ids{};
    Down down{};
    explicit Inputs(const std::filesystem::path& reference) {
        const auto n = read_checkpoint_data(reference, "ffn_norm-0");
        const auto i = read_checkpoint_data(reference, "ffn_moe_topk-0");
        const auto d = read_checkpoint_data(reference, "ffn_moe_down-0");
        if (n.metadata.type != GGML_TYPE_F32 || n.metadata.dimensions != std::array<int64_t, 4>{4096, 1, 1, 1} ||
            i.metadata.type != GGML_TYPE_I32 || i.metadata.dimensions != std::array<int64_t, 4>{6, 1, 1, 1} ||
            d.metadata.type != GGML_TYPE_F32 || d.metadata.dimensions != std::array<int64_t, 4>{4096, 6, 1, 1} ||
            n.data.size() != sizeof(norm) || i.data.size() != sizeof(ids) || d.data.size() != sizeof(down))
            throw std::runtime_error("mapped carrier requires actual layer-zero TG1 checkpoints");
        std::memcpy(norm.data(), n.data.data(), sizeof(norm));
        std::memcpy(ids.data(), i.data.data(), sizeof(ids));
        std::memcpy(down.data(), d.data.data(), sizeof(down));
        std::array<bool, 256> seen{};
        for (const auto id : ids) {
            if (id < 0 || id >= 256 || seen[id]) throw std::runtime_error("actual route ID contract differs");
            seen[id] = true;
        }
        for (const auto x : norm) if (!std::isfinite(x)) throw std::runtime_error("actual norm is nonfinite");
        for (const auto x : down) if (!std::isfinite(x)) throw std::runtime_error("actual Down is nonfinite");
    }
};
struct DeviceInputs {
    ggml_backend_t backend;
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *norm = nullptr, *ids = nullptr, *down = nullptr;
    explicit DeviceInputs(ggml_backend_t b, const Inputs& input) : backend(b) {
        context = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
        if (!context) throw std::runtime_error("mapped probe metadata allocation failed");
        norm = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4096);
        ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, 6);
        down = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4096, 6);
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!buffer) { ggml_free(context); context = nullptr; throw std::runtime_error("mapped probe GPU allocation failed"); }
        ggml_backend_tensor_set(norm, input.norm.data(), 0, sizeof(input.norm));
        ggml_backend_tensor_set(ids, input.ids.data(), 0, sizeof(input.ids));
        reset(input);
    }
    ~DeviceInputs() {
        ggml_backend_synchronize(backend);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
    void reset(const Inputs& input) { ggml_backend_tensor_set(down, input.down.data(), 0, sizeof(input.down)); }
    Down read() {
        Down result{};
        ggml_backend_tensor_get(down, result.data(), 0, sizeof(result));
        return result;
    }
};
struct Storage {
    MappedColdStorage value;
    Storage(ggml_backend_t backend, uint32_t timeout) { mapped_cold_create(backend, timeout, value); }
    ~Storage() { mapped_cold_destroy(value); }
};
struct HostBoundary {
    DeviceInputs& device;
    CpuExpertService& service;
    std::array<int32_t, 6> ids{}, compact_ids{};
    std::array<float, 4096> norm{};
    Down result{};
    uint64_t jobs = 0, experts = 0, service_ns = 0;
    void run(uint32_t mask) {
        // Same original route read on every legacy CPU-provider boundary.
        ggml_backend_tensor_get_async(device.backend, device.ids, ids.data(), 0, sizeof(ids));
        ggml_backend_synchronize(device.backend);
        if (!mask) return;
        ggml_backend_tensor_get_async(device.backend, device.norm, norm.data(), 0, sizeof(norm));
        ggml_backend_synchronize(device.backend);
        std::size_t count = 0;
        for (uint32_t rank = 0; rank < 6; ++rank)
            if (mask & (1u << rank)) compact_ids[count++] = ids[rank];
        const auto begin = Clock::now();
        service.compute(0, norm.data(), compact_ids.data(), count, result.data());
        service_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
        ++jobs; experts += count;
        std::size_t compact = 0;
        for (uint32_t rank = 0; rank < 6; ++rank)
            if (mask & (1u << rank)) {
                ggml_backend_tensor_set_async(device.backend, device.down, result.data() + 4096 * compact,
                                              rank * row_bytes, row_bytes);
                ++compact;
            }
    }
};
template<class Submit, class Drain>
std::array<double, 3> measure(Submit submit, Drain drain) {
    for (int warm = 0; warm < 3; ++warm) { submit(); drain(); }
    std::array<double, 3> samples{};
    for (auto& ms : samples) {
        const auto begin = Clock::now();
        for (int replay = 0; replay < 16; ++replay) { submit(); drain(); }
        ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count() / 16;
    }
    return samples;
}
double median(std::array<double, 3> samples) { std::sort(samples.begin(), samples.end()); return samples[1]; }
void write_samples(std::ostream& report, const char* name, const std::array<double, 3>& samples) {
    report << std::quoted(name) << ":[" << samples[0] << ',' << samples[1] << ',' << samples[2] << ']';
}
void write_binary(const std::filesystem::path& path, const Down& data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), sizeof(data));
    if (!out) throw std::runtime_error("mapped probe binary write failed");
}
int count_bits(uint32_t mask) {
    int count = 0;
    for (; mask; mask >>= 1) count += mask & 1;
    return count;
}
void run_case(DeviceInputs& device, CpuExpertService& service, const Inputs& input, uint32_t mask,
              const std::filesystem::path& directory, std::ostream& report, bool comma) {
    device.reset(input);
    HostBoundary control{device, service};
    const auto control_samples = measure([&] { control.run(mask); }, [&] { ggml_backend_synchronize(device.backend); });
    const auto expected = device.read();
    device.reset(input);
    Storage storage(device.backend, 500);
    std::array<double, 3> candidate_samples{};
    MappedColdMetrics metrics;
    {
        MappedColdWorker worker(service, storage.value);
        candidate_samples = measure([&] { mapped_cold_enqueue(storage.value, device.norm, device.ids, device.down, mask, 0); },
                                    [&] { mapped_cold_synchronize(storage.value); });
        metrics = worker.metrics();
    }
    const auto actual = device.read();
    const bool exact = std::memcmp(expected.data(), actual.data(), sizeof(actual)) == 0;
    const uint64_t expected_jobs = mask ? 51 : 0;
    const uint64_t expected_experts = expected_jobs * count_bits(mask);
    const uint32_t ready = mapped_cold_ready(storage.value.host);
    if (!exact || mapped_cold_status(storage.value) != 0 || metrics.jobs != expected_jobs ||
        metrics.experts != expected_experts || ready != expected_jobs)
        throw std::runtime_error("mapped carrier first actual layer output/protocol difference");
    write_binary(directory / ("mask" + std::to_string(mask) + "-control.bin"), expected);
    write_binary(directory / ("mask" + std::to_string(mask) + "-mapped.bin"), actual);
    if (comma) report << ",\n";
    const uint64_t norm_bytes = mask ? sizeof(input.norm) : 0;
    const uint64_t contribution_bytes = uint64_t(count_bits(mask)) * row_bytes;
    report << "{\"mask\":" << mask << ",\"cold_experts\":" << count_bits(mask)
           << ",\"output_bytes\":" << sizeof(actual) << ",\"complete_output_bit_exact\":true"
           << ",\"control_wall_ms\":" << median(control_samples)
           << ",\"mapped_wall_ms\":" << median(candidate_samples)
           << ",\"speedup\":" << median(control_samples) / median(candidate_samples) << ',';
    write_samples(report, "control_samples_ms", control_samples); report << ',';
    write_samples(report, "mapped_samples_ms", candidate_samples);
    report << ",\"control_cpu_service_ms_per_job\":" << (control.jobs ? double(control.service_ns) / control.jobs / 1e6 : 0)
           << ",\"mapped_cpu_service_ms_per_job\":" << (metrics.jobs ? double(metrics.service_ns) / metrics.jobs / 1e6 : 0)
           << ",\"control_api_d2h_bytes_per_replay\":" << sizeof(input.ids) + norm_bytes
           << ",\"control_api_h2d_bytes_per_replay\":" << contribution_bytes
           << ",\"mapped_api_h2d_d2h_bytes_per_replay\":0"
           << ",\"mapped_activation_and_id_payload_bytes_per_replay\":" << (mask ? norm_bytes + sizeof(input.ids) : 0)
           << ",\"mapped_contribution_payload_bytes_per_replay\":" << contribution_bytes
           << ",\"mapped_control_words\":\"nonzero mask: layer/mask, ready release, done release and bounded polling; mapped traffic is PCIe, not free\""
           << ",\"control_mid_layer_host_syncs_per_replay\":" << (mask ? 2 : 1)
           << ",\"mapped_mid_layer_host_syncs_per_replay\":0,\"mapped_gpu_launches_per_replay\":2"
           << ",\"common_measurement_end_drains_per_replay\":1,\"mapped_cpu_jobs\":" << metrics.jobs
           << ",\"mapped_cpu_experts\":" << metrics.experts << '}';
}
void negative_timeout(DeviceInputs& device, const Inputs& input, std::ostream& report) {
    device.reset(input);
    Storage storage(device.backend, 5); // No worker: it must fail without an accepted output.
    const auto start = Clock::now();
    mapped_cold_enqueue(storage.value, device.norm, device.ids, device.down, 1, 0);
    mapped_cold_synchronize(storage.value);
    const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const auto status = mapped_cold_status(storage.value);
    const auto output = device.read();
    if (status != 3 || elapsed > 1000 || !std::all_of(output.begin(), output.end(), [](float x) { return std::isnan(x); }))
        throw std::runtime_error("mapped carrier timeout did not reject the full output");
    report << "],\"negative_no_worker\":{\"fatal_status\":" << status
           << ",\"wall_ms\":" << elapsed << ",\"full_output_nan\":true}}\n";
}
int run(int argc, char** argv) {
    std::string model_path, reference, directory;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("mapped probe requires flag/value pairs");
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--model") model_path = value;
        else if (flag == "--reference-dir") reference = value;
        else if (flag == "--output-dir") directory = value;
        else throw std::runtime_error("unknown mapped probe option");
    }
    if (model_path.empty() || reference.empty() || directory.empty()) throw std::runtime_error("mapped probe paths required");
    std::filesystem::create_directories(directory);
    if (!std::filesystem::is_empty(directory)) throw std::runtime_error("mapped probe output must be empty");
    const Inputs input(reference);
    llama_backend_init();
    ggml_backend_dev_t devices[] = {nullptr};
    auto parameters = llama_model_default_params();
    parameters.devices = devices; parameters.n_gpu_layers = 0; parameters.split_mode = LLAMA_SPLIT_MODE_NONE;
    parameters.load_mode = LLAMA_LOAD_MODE_MMAP; parameters.lazy_mode = LLAMA_LAZY_MODE_ON;
    parameters.use_extra_bufts = false; parameters.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), parameters), llama_model_free);
    if (!model || model->arch != LLM_ARCH_DEEPSEEK4) throw std::runtime_error("mapped probe actual model unavailable");
    auto* device = ggml_backend_dev_by_name("CUDA0");
    if (!device) throw std::runtime_error("mapped probe CUDA0 unavailable");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    if (!backend) throw std::runtime_error("mapped probe CUDA backend unavailable");
    CpuExpertService service(*model, 16);
    DeviceInputs tensors(backend.get(), input);
    std::ofstream report(std::filesystem::path(directory) / "report.json");
    report << std::setprecision(10)
           << "{\"status\":\"passed\",\"timing_scope\":\"real GGUF layer0 TG1 cold boundary only; canonical CPU16 raw Down; three warmups and three16-replay wall windows; one common end drain per replay; no full-model speedup claim\",\"cases\":[\n";
    bool comma = false;
    for (const uint32_t mask : {0u, 1u, 7u, 63u}) {
        run_case(tensors, service, input, mask, directory, report, comma); comma = true;
    }
    negative_timeout(tensors, input, report);
    if (!report) throw std::runtime_error("mapped probe report write failed");
    std::cout << "{\"status\":\"passed\",\"real_layer_cases\":4,\"timeout_rejected\":true}\n";
    return 0;
}
}
int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
