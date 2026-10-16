#include "long_prefix_compare.hpp"
#include "long_prefix_dispatch.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace strata::deepseek_v4 {
PrefixReference prefix_read_reference(const std::filesystem::path& path) {
    if (std::filesystem::file_size(path) > 128 * 1024) throw std::runtime_error("prefix reference too large");
    std::ifstream input(path);
    std::string magic;
    int ni = 0, no = 0;
    input >> magic >> ni >> no;
    if (magic != "STRATA_DSV4_LONG_REFERENCE_V1" || ni != 8192 || no != 2048)
        throw std::runtime_error("invalid prefix reference");
    PrefixReference result;
    result.input.resize(ni); result.output.resize(no);
    for (auto& id : result.input) if (!(input >> id) || id < 0) throw std::runtime_error("invalid input ID");
    for (auto& id : result.output) if (!(input >> id) || id < 0) throw std::runtime_error("invalid output ID");
    std::string extra;
    if (input >> extra) throw std::runtime_error("unexpected prefix reference data");
    return result;
}
void prefix_write_difference(std::ostream& out, const PrefixDifference& d) {
    out << "{\"first_logit_difference_step\":" << d.step << ",\"first_token_difference_step\":" << d.token_step
        << ",\"differing_logits\":" << d.differing << ",\"max_abs\":" << d.max_abs
        << ",\"rms\":" << d.rms << ",\"expected_id\":" << d.expected << ",\"actual_id\":" << d.actual << '}';
}
PrefixDifference prefix_case(Runtime& runtime, PrefixReference& reference, const std::filesystem::path& output,
    const std::string& name, bool collect, int count, bool capture_candidate, PrefixDispatchSnapshots* snapshots, bool follow_actual_ids) {
    if (count < 1 || count > 2048) throw std::runtime_error("prefix count outside1..2048");
    if (follow_actual_ids && !capture_candidate) throw std::runtime_error("live reference requires capture");
    runtime.reset();
    std::string error;
    if (snapshots) snapshots->begin();
    if (!runtime.evaluate(reference.input, error)) throw std::runtime_error(error);
    if (snapshots) snapshots->end(runtime);
    PrefixDifference difference;
    const auto vocab = llama_vocab_n_tokens(runtime.vocab());
    std::vector<llama_token> actual;
    for (int step = 0; step < count; ++step) {
        llama_token id;
        try { id = runtime.greedy(true); }
        catch (const std::exception& failure) {
            // A cache timeout deliberately poisons logits. Preserve its cause
            // before the context dies, together with the actual input history.
            const auto* failed_logits = llama_get_logits_ith(runtime.context(), -1);
            if (failed_logits) {
                std::ofstream data(output / (name + "-failed.f32"), std::ios::binary);
                data.write(reinterpret_cast<const char*>(failed_logits), vocab * sizeof(float));
            }
            std::ofstream receipt(output / (name + "-failure.json"));
            receipt << "{\"failed_output_step\":" << step << ",\"completed_output_ids\":[";
            for (size_t i = 0; i < actual.size(); ++i) { if (i) receipt << ','; receipt << actual[i]; }
            receipt << "]}"; receipt.close();
            std::string cause = failure.what();
            try { verify_device_expert_cache(runtime.context()); }
            catch (const std::exception& cache_failure) { cause += "; "; cause += cache_failure.what(); }
            throw std::runtime_error("output step" + std::to_string(step) + ": " + cause);
        }
        actual.push_back(id);
        if (follow_actual_ids) reference.output[step] = id; // Next input and expected ID come from this live greedy run.
        const auto* logits = llama_get_logits_ith(runtime.context(), -1);
        if (!logits || vocab != 129280) throw std::runtime_error("prefix vocabulary/logits differ");
        if (step == 0) {
            std::ofstream first(output / (name + "-first.f32"), std::ios::binary);
            first.write(reinterpret_cast<const char*>(logits), vocab * sizeof(float));
            if (!first) throw std::runtime_error("first logits write failed");
        }
        if (collect || capture_candidate) reference.logits.emplace_back(logits, logits + vocab);
        else if (step < int(reference.logits.size())) {
            const auto& golden = reference.logits[step];
            double squares = 0;
            for (int k = 0; k < vocab; ++k) {
                if (!std::isfinite(logits[k]) || !std::isfinite(golden[k])) throw std::runtime_error("nonfinite prefix logits");
                if (std::memcmp(logits + k, golden.data() + k, sizeof(float))) {
                    ++difference.differing;
                    const double delta = double(logits[k]) - golden[k];
                    difference.max_abs = std::max(difference.max_abs, std::abs(delta));
                    squares += delta * delta;
                }
            }
            if (difference.differing) {
                difference.step = step;
                difference.rms = std::sqrt(squares / vocab);
                for (int copy = 0; copy < 2; ++copy) {
                    std::ofstream data(output / (name + (copy ? "-candidate.f32" : "-control.f32")), std::ios::binary);
                    data.write(reinterpret_cast<const char*>(copy ? logits : golden.data()), vocab * sizeof(float));
                    if (!data) throw std::runtime_error("prefix logits write failed");
                }
            }
        }
        if (id != reference.output[step]) {
            difference.token_step = step;
            difference.expected = reference.output[step]; difference.actual = id;

        }
        if (difference.step >= 0 || difference.token_step >= 0 || step + 1 == count) {
            std::ofstream data(output / (name + "-last.f32"), std::ios::binary);
            data.write(reinterpret_cast<const char*>(logits), vocab * sizeof(float));
            if (!data) throw std::runtime_error("last logits write failed");
            break;
        }
        if ((step + 1) % 32 == 0) {
            std::ofstream progress(output / "progress.json");
            progress << "{\"phase\":" << session_json(name) << ",\"completed_outputs\":" << step + 1 << '}';
        }
        if (step + 1 < count && !runtime.evaluate({reference.output[step]}, error)) throw std::runtime_error(error);
    }
    verify_device_expert_cache(runtime.context());
    std::ofstream receipt(output / (name + "-prefix.json"));
    receipt << "{\"difference\":"; prefix_write_difference(receipt, difference);
    receipt << ",\"output_ids\":[";
    for (size_t i = 0; i < actual.size(); ++i) { if (i) receipt << ','; receipt << actual[i]; }
    receipt << "]}\n";
    if (!receipt) throw std::runtime_error("prefix receipt write failed");
    if (collect && !capture_candidate && difference.token_step >= 0)
        throw std::runtime_error("new control differs from frozen receipt at step" + std::to_string(difference.token_step));
    return difference;
}
}
