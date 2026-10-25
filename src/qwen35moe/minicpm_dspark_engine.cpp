#include "strata/qwen35moe/runtime.hpp"
#include "strata/models/native_options.hpp"
#include "strata/models/request_sampling.hpp"

#include <cuda_profiler_api.h>
#include "native_step_observation.hpp"

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

// Process-local CUDA-profiler ranges; external tooling/CLI options remain separate.
bool cuda_trace_ranges_enabled() {
    const char* value = std::getenv("STRATA_QWEN35_CUDA_TRACE_RANGES");
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument("STRATA_QWEN35_CUDA_TRACE_RANGES must be unset, 0 or 1");
}

class CudaTraceRanges {
public:
    explicit CudaTraceRanges(bool enabled) : enabled_(enabled) {}
    ~CudaTraceRanges() { stop("scope-exit"); }
    CudaTraceRanges(const CudaTraceRanges&) = delete;
    CudaTraceRanges& operator=(const CudaTraceRanges&) = delete;
    void select_slot(int slot) { if (armed_ && slot != slot_) stop("session-switch"); }
    void begin_prefill(int slot, size_t tokens, bool eligible) {
        if (!enabled_) return;
        stop("prepare");
        const int index = tokens == 8192 ? 0 : tokens == 30000 ? 1 : -1;
        if (failed_ || !eligible || index < 0 || claimed_[size_t(index)]) return;
        claimed_[size_t(index)] = true; armed_ = prefill_ = true; slot_ = slot; tokens_ = tokens;
        warm_ = captured_ = errors_ = 0; start_code_ = stop_code_ = -1;
        start();
    }
    void end_prefill() {
        if (!armed_ || !prefill_) return;
        const int slot = slot_;
        stop("prefill-complete");
        if (failed_) return;
        armed_ = true; prefill_ = false; slot_ = slot;
        warm_ = captured_ = errors_ = 0; start_code_ = stop_code_ = -1;
        log("arm", "await-warmup");
    }
    void before_step(int slot) {
        if (!armed_ || prefill_) return;
        if (slot != slot_) { stop("session-switch"); return; }
        if (warm_ == 8 && !running_) start();
    }
    void after_step(int slot) {
        if (!armed_ || prefill_ || slot != slot_) return;
        if (!running_) { ++warm_; return; }
        if (++captured_ == 8) stop("window-complete");
    }
    void stop(const char* reason) noexcept {
        if (!armed_ && !running_) return;
        if (running_) {
            stop_code_ = int(cudaProfilerStop());
            if (stop_code_ != int(cudaSuccess)) { ++errors_; failed_ = true; }
        }
        log("end", reason); running_ = armed_ = false; slot_ = -1;
    }
private:
    void start() noexcept {
        start_code_ = int(cudaProfilerStart()); running_ = true;
        if (start_code_ != int(cudaSuccess)) {
            ++errors_; failed_ = true; stop("start-error"); return;
        }
        log("begin", prefill_ ? "prefill" : "warmup-complete");
    }
    void log(const char* event, const char* reason) const noexcept {
        std::fprintf(stderr, "qwen35_cuda_trace_range event=%s phase=%s reason=%s slot=%d "
            "input_tokens=%zu warm_steps=%u requested_steps=%u captured_steps=%u "
            "cuda_errors=%u start_code=%d stop_code=%d diagnostic=1 fair_PP_TG=0 GPU_busy_measured=0\n",
            event, prefill_ ? "PP" : "TG", reason, slot_, tokens_, warm_, prefill_ ? 0u : 8u,
            captured_, errors_, start_code_, stop_code_);
        std::fflush(stderr);
    }
    bool enabled_, armed_ = false, running_ = false, prefill_ = false, failed_ = false;
    std::array<bool, 2> claimed_{false, false};
    int slot_ = -1, start_code_ = -1, stop_code_ = -1;
    size_t tokens_ = 0;
    unsigned warm_ = 0, captured_ = 0, errors_ = 0;
};

void request(Runtime& runtime, DsparkVerifyState& verify, const std::string& aux_path,
             const std::string& line, std::array<int, 2>& positions,
             std::array<int, 2>& pendings, std::array<bool, 2>& active,
             std::array<SlotSampling, 2>& sampling,
             std::array<std::vector<int>, 2>& snapshots, std::array<bool, 2>& resumables,
             bool dspark_enabled, bool& decode_started, CudaTraceRanges& trace) {
    std::istringstream input(line);
    std::string command, csv, extra;
    if (!(input >> command)) throw std::invalid_argument("empty DSpark command");
    int slot = -1;
    if (!(input >> slot) || slot < 0 || slot >= runtime.session_capacity())
        throw std::invalid_argument("invalid MiniCPM request slot");
    trace.select_slot(slot);
    int& position = positions[size_t(slot)];
    int& pending = pendings[size_t(slot)];
    auto& sampler = sampling[size_t(slot)];
    if (command == "CONFIG") {
        trace.stop("config");
        if (active[size_t(slot)]) throw std::invalid_argument("MiniCPM slot is active");
        sampler = SlotSampling{};
        sampler.configure(input, runtime.max_context());
        std::printf("CONFIGURED slot=%d\n", slot); std::fflush(stdout);
        return;
    }
    if (command == "PREPARE" || command == "PREPAREV") {
        trace.stop("prepare-command");
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
                resumables[size_t(id)] = false;
                snapshots[size_t(id)].clear();
            }
        }
        // A repeated prefix keeps its K/V and GDN state; the linear recurrence
        // cannot rewind per token, so only a full-prefix match resumes.
        int resume = 0;
        auto& snapshot = snapshots[size_t(slot)];
        if (command == "PREPARE" && resumables[size_t(slot)] && !snapshot.empty() &&
            runtime.reserved_context() >= int(prompt.size()) + max_new) {
            const size_t limit = (std::min)(snapshot.size(), prompt.size());
            size_t common = 0;
            while (common < limit && snapshot[common] == prompt[common]) ++common;
            if (common == snapshot.size() && common < prompt.size()) resume = int(common);
            else if (runtime.prefill_checkpoint() > 0 &&
                     size_t(runtime.prefill_checkpoint()) <= common) resume = runtime.prefill_checkpoint();
        }
        if (decode_started) { runtime.finish_decode(); decode_started = false; }
        runtime.select_session(slot);
        runtime.reset_session(resume > 0);
        if (resume > 0 && resume == runtime.prefill_checkpoint()) runtime.restore_prefill_checkpoint();
        resumables[size_t(slot)] = false;
        Metrics metrics;
        const auto started = Clock::now();
        const int chunk = runtime.preferred_prefill_capacity(int(prompt.size()));
        runtime.reserve_context(int(prompt.size()) + max_new, metrics,
                                (std::min)(chunk, int(prompt.size())));
        runtime.prepare_batch((std::max)(16, (std::min)(chunk, int(prompt.size()))));
        // The visual rows are bounded by the context reserved just above.
        if (command == "PREPAREV") runtime.set_visual_input(std::move(visual));
        const bool draft_request = dspark_enabled && !sampler.enabled();
        PromptAux aux(draft_request ? prompt.size() : 0);
        if (draft_request) runtime.set_checkpoint_sink(&aux);
        SinkGuard guard{runtime};
        // Live PP progress: the long prefill reports layer completions at
        // existing route drains, so no extra device events are added.
        struct PpProgress {
            Clock::time_point start;
            int tokens = 0;
            int step = -1;
        } pp{Clock::now(), int(prompt.size())};
        auto pp_report = [](void* context, int completed, int total) {
            auto& s = *static_cast<PpProgress*>(context);
            if (total < 1) return;
            const int step = completed * 20 / total;
            if (step == s.step) return;
            s.step = step;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - s.start).count();
            const int done = int(static_cast<long long>(s.tokens) * completed / total);
            std::printf("PP %d %d %.0f %.1f\n", done, s.tokens, ms, ms > 0 ? 1000.0 * done / ms : 0.0);
            std::fflush(stdout);
        };
        static const bool pp_progress = [] {
            const char* flag = std::getenv("STRATA_QWEN35_PP_PROGRESS");
            return !flag || flag[0] != '0';
        }();
        const PrefillObserver observer{pp_progress ? &pp : nullptr, pp_report};
        trace.begin_prefill(slot, prompt.size(), command == "PREPARE" && !draft_request);
        const int tail = int(prompt.size()) - resume;
        // A client that drops generated reasoning leaves the next prompt a few
        // template tokens short of this one's end. Stop just before the render
        // tail so the recurrent snapshot stays at or below that common prefix.
        const int split = (resume == 0 && tail > 128) ? tail - 16 : 0;
        int anchor = 0;
        if (split > 0) {
            anchor = split > chunk
                ? runtime.run_prefill_long(prompt.data(), split, metrics, chunk, &observer, 0)
                : runtime.run_prefill(prompt.data(), split, "", metrics, &observer, 0);
            runtime.save_prefill_checkpoint(split);
            anchor = runtime.run_prefill(prompt.data() + split, tail - split, "", metrics, nullptr, split);
        } else {
            anchor = tail > chunk
                ? runtime.run_prefill_long(prompt.data() + resume, tail, metrics, chunk, &observer, resume)
                : runtime.run_prefill(prompt.data() + resume, tail, "", metrics, &observer, resume);
        }
        trace.end_prefill();
        runtime.set_checkpoint_sink(nullptr);
        sampler.prepare(prompt);
        const int selected = sampler.enabled() ? sampler.pick(runtime, metrics) : anchor;
        if (draft_request) write_aux(aux_path, aux.values);
        runtime.start_decode(&metrics);
        decode_started = true;
        position = int(prompt.size());
        pending = selected;
        active[size_t(slot)] = true;
        snapshot.assign(prompt.begin(), prompt.end());
        snapshot.push_back(selected);
        if (const char* profile = std::getenv("STRATA_MINICPM_PP_PROFILE");
            profile && std::string(profile) == "1") {
            std::array<char, 129> pinned_error{};
            const auto& pin_error = runtime.pinned_expert_error();
            const size_t error_size = (std::min)(pin_error.size(), pinned_error.size() - 1);
            for (size_t i = 0; i < error_size; ++i) {
                const auto c = static_cast<unsigned char>(pin_error[i]);
                const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                  (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' || c == ':';
                pinned_error[i] = safe ? static_cast<char>(c) : '_';
            }
            std::fprintf(stderr,
                "minicpm_pp_profile tokens=%zu batch=%d slots=%d cache_slots=%zu "
                "dense_ms=%.3f route_ms=%.3f copy_ms=%.3f moe_ms=%.3f "
                "expert_h2d=%llu expert_h2d_calls=%llu hits=%llu misses=%llu "
                "expert_launches=%llu sync=%llu "
                "gdn_ms=%.3f attention_ms=%.3f gu_ms=%.3f down_ms=%.3f shared_ms=%.3f "
                "address_ms=%.3f pinned_layers=%zu pinned_bytes=%zu pinned_error=%s\n",
                prompt.size(), chunk, runtime.session_capacity(), runtime.expert_cache_slots(),
                metrics.pp_dense_event_ms, metrics.pp_route_event_ms,
                metrics.pp_copy_event_ms, metrics.pp_moe_event_ms,
                static_cast<unsigned long long>(metrics.expert_h2d),
                static_cast<unsigned long long>(metrics.expert_h2d_calls),
                static_cast<unsigned long long>(metrics.expert_cache_hits),
                static_cast<unsigned long long>(metrics.expert_cache_misses),
                static_cast<unsigned long long>(metrics.expert_launches),
                static_cast<unsigned long long>(metrics.explicit_syncs),
                metrics.pp_gdn_event_ms, metrics.pp_attention_event_ms,
                metrics.pp_gu_event_ms, metrics.pp_down_event_ms, metrics.pp_shared_event_ms,
                metrics.pp_address_ms, runtime.pinned_expert_layers(), runtime.pinned_expert_bytes(),
                error_size ? pinned_error.data() : "none");
            std::fflush(stderr);
        }
        std::printf("PREFILL anchor=%d tokens=%zu pp_ms=%.3f aux=%zu cache_slots=%zu pp_resume=%d\n",
                    selected, prompt.size(), elapsed_ms(started),
                    aux.values.size() * sizeof(float), runtime.expert_cache_slots(), resume);
        std::fflush(stdout);
        return;
    }
    if (command == "RELEASE") {
        trace.stop("release");
        if (input >> extra) throw std::invalid_argument("unexpected RELEASE field");
        const bool keep = !snapshots[size_t(slot)].empty() && position > 0;
        if (decode_started) { runtime.finish_decode(); decode_started = false; }
        runtime.select_session(slot); runtime.reset_session(keep);
        resumables[size_t(slot)] = keep;
        if (!keep) snapshots[size_t(slot)].clear();
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
        trace.before_step(slot);
        const int greedy = runtime.run_token(anchor, position, true, metrics);
        sampler.remember(anchor);
        const int next = sampler.enabled() ? sampler.pick(runtime, metrics) : greedy;
        ++position;
        pending = next;
        snapshots[size_t(slot)].push_back(next);
        trace.after_step(slot);
        print_native_step(next, elapsed_ms(started), metrics);
        std::fflush(stdout);
        return;
    }
    if (command == "VERIFY") {
        trace.stop("verify");
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
        snapshots[size_t(slot)].insert(snapshots[size_t(slot)].end(),
                                       result.committed_ids.begin(), result.committed_ids.end());
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
        const bool trace_enabled = cuda_trace_ranges_enabled();
        const bool dspark_enabled = argc == 6 && std::string(argv[5]) == "1";
#ifdef _WIN32
        _putenv_s("STRATA_QWEN35_SESSIONS", argv[4]);
#else
        setenv("STRATA_QWEN35_SESSIONS", argv[4], 1);
#endif
        Runtime runtime(argv[1], std::stoi(argv[2]), true);
        CudaTraceRanges trace(trace_enabled);
        DsparkVerifyState verify;
        if (dspark_enabled) runtime.prepare_dspark_verify(verify);
        std::array<int, 2> positions{-1, -1}, pendings{-1, -1};
        std::array<bool, 2> active{false, false};
        std::array<SlotSampling, 2> sampling{};
        std::array<std::vector<int>, 2> snapshots{};
        std::array<bool, 2> resumables{false, false};
        bool decode_started = false;
        std::printf("READY slots=%d\n", runtime.session_capacity()); std::fflush(stdout);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") { trace.stop("quit"); break; }
            try { request(runtime, verify, argv[3], line, positions, pendings, active, sampling,
                          snapshots, resumables, dspark_enabled, decode_started, trace); }
            catch (const std::exception& error) {
                trace.stop("exception");
                std::fprintf(stderr, "DSpark request: %s\n", error.what());
                std::printf("ERROR %s\n", error.what()); std::fflush(stdout);
            }
        }
        trace.stop("eof");
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
