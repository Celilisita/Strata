#include "strata/qwen35moe/runtime.hpp"
#include "strata/models/native_options.hpp"
#include "strata/models/request_sampling.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <vector>

namespace {
using namespace strata::qwen35moe;
using Clock = std::chrono::steady_clock;

std::vector<int> parse_ids(const std::string& csv) {
    if (csv.empty() || csv.back() == ',') throw std::invalid_argument("empty ID list");
    std::vector<int> ids;
    size_t first = 0;
    while (first < csv.size()) {
        const size_t comma = csv.find(',', first);
        const std::string piece = csv.substr(first, comma - first);
        size_t used = 0;
        const int value = std::stoi(piece, &used);
        if (piece.empty() || used != piece.size()) throw std::invalid_argument("invalid ID");
        ids.push_back(value);
        if (comma == std::string::npos) break;
        first = comma + 1;
    }
    return ids;
}

struct VisualPrompt {
    std::vector<int> ids;
    VisualInput input;
};

VisualPrompt read_visual_prompt(const std::string& path, int max_context) {
    std::ifstream file(path, std::ios::binary);
    int32_t header[3]{};
    if (!file.read(reinterpret_cast<char*>(header), sizeof(header)) ||
        header[0] != 0x31505644 || header[1] < 2 || header[1] > max_context ||
        header[2] < 1 || header[2] > 64)
        throw std::invalid_argument("invalid MiniCPM visual prompt header");
    VisualPrompt prompt;
    prompt.ids.resize(size_t(header[1]));
    prompt.input.positions.resize(size_t(header[1]));
    if (!file.read(reinterpret_cast<char*>(prompt.ids.data()),
                   std::streamsize(prompt.ids.size() * sizeof(int32_t))) ||
        !file.read(reinterpret_cast<char*>(prompt.input.positions.data()),
                   std::streamsize(prompt.input.positions.size() * 3 * sizeof(int32_t))))
        throw std::invalid_argument("truncated MiniCPM visual prompt");
    int previous_end = 0;
    for (int i = 0; i < header[2]; ++i) {
        int32_t span[2]{};
        if (!file.read(reinterpret_cast<char*>(span), sizeof(span)) ||
            span[0] < previous_end || span[1] < 1 ||
            span[0] > header[1] - span[1])
            throw std::invalid_argument("invalid MiniCPM visual span");
        ImageEmbedding image{span[1], 1, 2048,
                             std::vector<float>(size_t(span[1]) * 2048)};
        if (!file.read(reinterpret_cast<char*>(image.rows.data()),
                       std::streamsize(image.rows.size() * sizeof(float))))
            throw std::invalid_argument("truncated MiniCPM visual embeddings");
        prompt.input.spans.push_back({span[0], std::move(image)});
        previous_end = span[0] + span[1];
    }
    if (file.peek() != std::char_traits<char>::eof())
        throw std::invalid_argument("MiniCPM visual prompt has trailing data");
    return prompt;
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void write_aux(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.write(reinterpret_cast<const char*>(values.data()),
                    std::streamsize(values.size() * sizeof(float))))
        throw std::runtime_error("cannot write DSpark auxiliary states");
}

class PromptAux final : public CheckpointSink {
public:
    explicit PromptAux(size_t count) : count_(count), values(count * 5 * 2048) {}
    void layer(int position, size_t layer, const float* hidden,
               cudaStream_t stream, Metrics&) override {
        constexpr std::array<size_t, 5> selected{3, 11, 19, 27, 35};
        for (size_t channel = 0; channel < selected.size(); ++channel) {
            if (layer != selected[channel]) continue;
            if (position < 0 || size_t(position) >= count_)
                throw std::out_of_range("DSpark prompt auxiliary position");
            cuda_check(cudaMemcpyAsync(values.data() + (size_t(position) * 5 + channel) * 2048,
                                       hidden, 2048 * sizeof(float), cudaMemcpyDeviceToHost, stream),
                       "capture DSpark prompt hidden state");
            return;
        }
    }
    void attention(int, const float*, const float*, const float*,
                   cudaStream_t, Metrics&) override {}
    size_t count_;
    std::vector<float> values;
};

struct SinkGuard {
    Runtime& runtime;
    ~SinkGuard() { runtime.set_checkpoint_sink(nullptr); }
};

struct SlotSampling : strata::models::RequestSampling {
    int pick(Runtime& runtime, Metrics& metrics) {
        const int token = runtime.sample_last_logits(params, history, metrics);
        ++params.counter;
        return token;
    }
};

void request(Runtime& runtime, DsparkVerifyState& verify, const std::string& aux_path,
             const std::string& line, std::array<int, 2>& positions,
             std::array<int, 2>& pendings, std::array<bool, 2>& active,
             std::array<SlotSampling, 2>& sampling,
             bool dspark_enabled, bool& decode_started) {
    std::istringstream input(line);
    std::string command, csv, extra;
    if (!(input >> command)) throw std::invalid_argument("empty DSpark command");
    int slot = -1;
    if (!(input >> slot) || slot < 0 || slot >= runtime.session_capacity())
        throw std::invalid_argument("invalid MiniCPM request slot");
    int& position = positions[size_t(slot)];
    int& pending = pendings[size_t(slot)];
    auto& sampler = sampling[size_t(slot)];
    if (command == "CONFIG") {
        if (active[size_t(slot)]) throw std::invalid_argument("MiniCPM slot is active");
        sampler = SlotSampling{};
        sampler.configure(input, runtime.max_context());
        std::printf("CONFIGURED slot=%d\n", slot); std::fflush(stdout);
        return;
    }
    if (command == "PREPARE" || command == "PREPAREV") {
        int max_new = 0;
        if (!(input >> max_new >> csv) || input >> extra)
            throw std::invalid_argument("expected PREPARE max_new prompt_ids");
        std::vector<int> prompt;
        VisualInput visual;
        if (command == "PREPAREV") {
            auto loaded = read_visual_prompt(csv, runtime.max_context());
            prompt = std::move(loaded.ids);
            visual = std::move(loaded.input);
        } else prompt = parse_ids(csv);
        if (prompt.size() < 2 || max_new < 1 ||
            prompt.size() + size_t(max_new) > size_t(runtime.max_context()))
            throw std::invalid_argument("DSpark prompt or context length");
        if (active[size_t(slot)]) throw std::invalid_argument("MiniCPM slot already active");
        if (std::any_of(prompt.begin(), prompt.end(), [&](int id) { return id < 0 || id >= int(runtime.spec().n_vocab); }))
            throw std::invalid_argument("MiniCPM token outside vocabulary");
        const int other = slot ^ 1;
        if (runtime.session_capacity() == 2 && active[size_t(other)] &&
            int(prompt.size()) + max_new > runtime.reserved_context()) {
            std::puts("BUSY"); std::fflush(stdout); return;
        }
        if (runtime.session_capacity() == 2 && !active[size_t(other)] &&
            int(prompt.size()) + max_new > runtime.reserved_context()) {
            if (decode_started) { runtime.finish_decode(); decode_started = false; }
            for (int id = 0; id < 2; ++id) {
                runtime.select_session(id);
                runtime.reset_session();
            }
        }
        if (decode_started) { runtime.finish_decode(); decode_started = false; }
        runtime.select_session(slot);
        runtime.reset_session();
        if (command == "PREPAREV") runtime.set_visual_input(std::move(visual));
        Metrics metrics;
        const auto started = Clock::now();
        const int chunk = runtime.preferred_prefill_capacity(int(prompt.size()));
        runtime.reserve_context(int(prompt.size()) + max_new, metrics,
                                (std::min)(chunk, int(prompt.size())));
        runtime.prepare_batch((std::max)(16, (std::min)(chunk, int(prompt.size()))));
        PromptAux aux(dspark_enabled ? prompt.size() : 0);
        if (dspark_enabled) runtime.set_checkpoint_sink(&aux);
        SinkGuard guard{runtime};
        const int anchor = prompt.size() > size_t(chunk)
            ? runtime.run_prefill_long(prompt.data(), int(prompt.size()), metrics, chunk)
            : runtime.run_prefill(prompt.data(), int(prompt.size()), "", metrics);
        runtime.set_checkpoint_sink(nullptr);
        sampler.prepare(prompt);
        const int selected = sampler.enabled() ? sampler.pick(runtime, metrics) : anchor;
        if (dspark_enabled) write_aux(aux_path, aux.values);
        runtime.start_decode(&metrics);
        decode_started = true;
        position = int(prompt.size());
        pending = selected;
        active[size_t(slot)] = true;
        if (const char* profile = std::getenv("STRATA_MINICPM_PP_PROFILE");
            profile && std::string(profile) == "1") {
            std::fprintf(stderr,
                "minicpm_pp_profile tokens=%zu batch=%d slots=%d cache_slots=%zu "
                "dense_ms=%.3f route_ms=%.3f copy_ms=%.3f moe_ms=%.3f "
                "expert_h2d=%llu expert_h2d_calls=%llu hits=%llu misses=%llu "
                "expert_launches=%llu sync=%llu\n",
                prompt.size(), chunk, runtime.session_capacity(), runtime.expert_cache_slots(),
                metrics.pp_dense_event_ms, metrics.pp_route_event_ms,
                metrics.pp_copy_event_ms, metrics.pp_moe_event_ms,
                static_cast<unsigned long long>(metrics.expert_h2d),
                static_cast<unsigned long long>(metrics.expert_h2d_calls),
                static_cast<unsigned long long>(metrics.expert_cache_hits),
                static_cast<unsigned long long>(metrics.expert_cache_misses),
                static_cast<unsigned long long>(metrics.expert_launches),
                static_cast<unsigned long long>(metrics.explicit_syncs));
            std::fflush(stderr);
        }
        std::printf("PREFILL anchor=%d tokens=%zu pp_ms=%.3f aux=%zu cache_slots=%zu\n",
                    selected, prompt.size(), elapsed_ms(started),
                    aux.values.size() * sizeof(float), runtime.expert_cache_slots());
        std::fflush(stdout);
        return;
    }
    if (command == "RELEASE") {
        if (input >> extra) throw std::invalid_argument("unexpected RELEASE field");
        if (decode_started) { runtime.finish_decode(); decode_started = false; }
        runtime.select_session(slot); runtime.reset_session();
        position = pending = -1;
        active[size_t(slot)] = false;
        sampler = SlotSampling{};
        std::printf("RELEASED slot=%d\n", slot); std::fflush(stdout);
        return;
    }
    if (command == "STEP") {
        int anchor = -1;
        if (!(input >> anchor) || input >> extra || !active[size_t(slot)] ||
            anchor != pending || position < 0 || position >= runtime.max_context())
            throw std::invalid_argument("expected STEP pending_anchor");
        runtime.select_session(slot);
        Metrics metrics;
        const auto started = Clock::now();
        if (!decode_started) { runtime.start_decode(&metrics); decode_started = true; }
        const int greedy = runtime.run_token(anchor, position, true, metrics);
        sampler.remember(anchor);
        const int next = sampler.enabled() ? sampler.pick(runtime, metrics) : greedy;
        ++position;
        pending = next;
        std::printf("STEPPED next=%d ms=%.3f\n", next, elapsed_ms(started));
        std::fflush(stdout);
        return;
    }
    if (command == "VERIFY") {
        if (!dspark_enabled) throw std::invalid_argument("DSpark verification is disabled");
        if (sampler.enabled())
            throw std::invalid_argument("DSpark verification requires greedy sampling");
        int anchor = -1;
        if (!(input >> anchor >> csv) || input >> extra || !active[size_t(slot)] ||
            anchor != pending || position < 0)
            throw std::invalid_argument("expected VERIFY pending_anchor proposal_ids");
        const auto draft = parse_ids(csv);
        if (draft.empty() || draft.size() > 15)
            throw std::invalid_argument("DSpark block must have 1..15 proposals");
        std::vector<int> block{anchor};
        block.insert(block.end(), draft.begin(), draft.end());
        Metrics metrics;
        const auto started = Clock::now();
        runtime.select_session(slot);
        if (!decode_started) { runtime.start_decode(&metrics); decode_started = true; }
        const auto result = runtime.verify_dspark_block(verify, block, position, metrics);
        write_aux(aux_path, result.auxiliary);
        position += int(result.committed_ids.size());
        pending = result.next;
        std::printf("VERIFIED ids=");
        for (size_t i = 0; i < result.committed_ids.size(); ++i)
            std::printf("%s%d", i ? "," : "", result.committed_ids[i]);
        std::printf(" next=%d verified=%d ms=%.3f aux=%zu rollback=%zu\n",
                    result.next, result.verified, elapsed_ms(started),
                    result.auxiliary.size() * sizeof(float), metrics.rollback_bytes);
        std::fflush(stdout);
        return;
    }
    throw std::invalid_argument("unknown DSpark command");
}
}

static int minicpm_protocol(int argc, char** argv) {
    if ((argc != 5 && argc != 6) ||
        (std::string(argv[4]) != "1" && std::string(argv[4]) != "2") ||
        (argc == 6 && std::string(argv[5]) != "0" && std::string(argv[5]) != "1")) {
        std::fprintf(stderr, "usage: strata-minicpm-dspark-engine MODEL.gguf CONTEXT AUX.f32 SLOTS [DSPARK]\n");
        return 2;
    }
    try {
        const bool dspark_enabled = argc == 6 && std::string(argv[5]) == "1";
#ifdef _WIN32
        _putenv_s("STRATA_QWEN35_SESSIONS", argv[4]);
#else
        setenv("STRATA_QWEN35_SESSIONS", argv[4], 1);
#endif
        Runtime runtime(argv[1], std::stoi(argv[2]), true);
        DsparkVerifyState verify;
        if (dspark_enabled) runtime.prepare_dspark_verify(verify);
        std::array<int, 2> positions{-1, -1}, pendings{-1, -1};
        std::array<bool, 2> active{false, false};
        std::array<SlotSampling, 2> sampling{};
        bool decode_started = false;
        std::printf("READY slots=%d\n", runtime.session_capacity()); std::fflush(stdout);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            try { request(runtime, verify, argv[3], line, positions, pendings, active, sampling,
                          dspark_enabled, decode_started); }
            catch (const std::exception& error) {
                std::fprintf(stderr, "DSpark request: %s\n", error.what());
                std::printf("ERROR %s\n", error.what()); std::fflush(stdout);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "DSpark engine: %s\n", error.what());
        return 1;
    }
}

int strata_qwen35_moe_serve(int argc, char** argv) {
    try {
        auto o = strata::models::NativeOptions::parse(argc, argv);
        std::string context = std::to_string(o.context), slots = std::to_string(o.slots);
        std::string enabled = o.dspark ? "1" : "0";
        char* args[] = {argv[0], o.model.data(), context.data(), o.aux.data(), slots.data(), enabled.data()};
        return minicpm_protocol(6, args);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MiniCPM configuration: %s\n", error.what()); return 2;
    }
}
