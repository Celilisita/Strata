#include "runtime.hpp"
#include "options.hpp"
#include "device_cache.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::deepseek_v4::Runtime;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool parse_ids(const std::string& value, std::vector<llama_token>& ids, bool image) {
    ids.clear();
    std::istringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) return false;
        char* end = nullptr;
        const long token = std::strtol(item.c_str(), &end, 10);
        if (!end || *end || token < (image ? -1 : 0) ||
            token > std::numeric_limits<llama_token>::max()) return false;
        ids.push_back(static_cast<llama_token>(token));
    }
    return !ids.empty();
}

struct Request {
    int limit = 0;
    bool ignore_eog = false;
    double temperature = 0.0;
    double top_p = 1.0;
    double min_p = 0.0;
    int top_k = 64;
    uint64_t seed = 0;
    std::string embeddings_path;
    std::vector<llama_token> ids;
};

bool parse_number(const std::string& value, double& result) {
    char* end = nullptr;
    result = std::strtod(value.c_str(), &end);
    return end != value.c_str() && *end == '\0' && std::isfinite(result);
}

llama_token sample_logits(Runtime& runtime, const Request& request, std::mt19937_64& rng) {
    const float* logits = llama_get_logits_ith(runtime.context(), -1);
    if (!logits) throw std::runtime_error("no output logits are available");
    const int32_t count = llama_vocab_n_tokens(runtime.vocab());
    if (count <= 0) throw std::runtime_error("model vocabulary is empty");
    struct Candidate { float logit; llama_token id; };
    const auto higher = [](const Candidate& a, const Candidate& b) { return a.logit > b.logit; };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(higher)> best(higher);
    for (int32_t id = 0; id < count; ++id) {
        const float logit = logits[id];
        if (std::isnan(logit) || logit == std::numeric_limits<float>::infinity())
            throw std::runtime_error("non-finite output logits");
        if (!std::isfinite(logit) || (request.ignore_eog && runtime.is_eog(id))) continue;
        if (best.size() < size_t(request.top_k)) best.push({logit, id});
        else if (logit > best.top().logit) { best.pop(); best.push({logit, id}); }
    }
    if (best.empty()) throw std::runtime_error("no finite output logits");
    std::vector<Candidate> candidates;
    candidates.reserve(best.size());
    while (!best.empty()) { candidates.push_back(best.top()); best.pop(); }
    std::sort(candidates.begin(), candidates.end(), higher);
    std::vector<double> weights;
    weights.reserve(candidates.size());
    double total = 0.0;
    for (const auto& item : candidates) {
        const double weight = std::exp((double(item.logit) - candidates.front().logit) / request.temperature);
        if (weight < request.min_p) break;
        weights.push_back(weight);
        total += weight;
    }
    if (weights.empty()) throw std::runtime_error("sampling removed every candidate");
    double kept = 0.0;
    size_t limit = 0;
    do { kept += weights[limit++]; } while (limit < weights.size() && kept / total < request.top_p);
    std::uniform_real_distribution<double> draw(0.0, kept);
    const double target = draw(rng);
    double cumulative = 0.0;
    llama_token chosen = candidates[limit - 1].id;
    for (size_t i = 0; i < limit; ++i) {
        cumulative += weights[i];
        if (target < cumulative) { chosen = candidates[i].id; break; }
    }
    strata::deepseek_v4::idle_device_expert_cache(runtime.context());
    return chosen;
}

bool parse_request(const std::string& line, Request& out, std::string& error) {
    const bool image = line.rfind("GENI ", 0) == 0;
    if (!image && line.rfind("GEN ", 0) != 0) {
        error = "expected GEN <max_new> <ids> or GENI <max_new> <file> <ids>";
        return false;
    }
    std::istringstream stream(line.substr(image ? 5 : 4));
    if (!(stream >> out.limit) || out.limit < 1) {
        error = "bad generation limit";
        return false;
    }
    std::string word;
    std::string id_text;
    while (stream >> word) {
        const auto equals = word.find('=');
        if (equals == std::string::npos) {
            if (image) {
                out.embeddings_path = word;
                stream >> id_text;
            } else {
                id_text = word;
            }
            break;
        }
        if (word.substr(0, equals) == "ignore_eog") {
            const auto value = word.substr(equals + 1);
            if (value != "0" && value != "1") {
                error = "ignore_eog must be 0 or 1";
                return false;
            }
            out.ignore_eog = value == "1";
        } else {
            const std::string key = word.substr(0, equals);
            const std::string value = word.substr(equals + 1);
            double number = 0.0;
            if (key == "temperature" || key == "top_p" || key == "min_p") {
                if (!parse_number(value, number) ||
                    (key == "temperature" && number < 0.0) ||
                    (key == "top_p" && (number <= 0.0 || number > 1.0)) ||
                    (key == "min_p" && (number < 0.0 || number > 1.0))) {
                    error = "invalid sampling value for " + key;
                    return false;
                }
                if (key == "temperature") out.temperature = number;
                else if (key == "top_p") out.top_p = number;
                else out.min_p = number;
            } else if (key == "top_k") {
                char* end = nullptr;
                const long parsed = std::strtol(value.c_str(), &end, 10);
                if (end == value.c_str() || *end || parsed < 1 || parsed > 64) {
                    error = "top_k must be in 1..64";
                    return false;
                }
                out.top_k = int(parsed);
            } else if (key == "seed") {
                char* end = nullptr;
                const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
                if (end == value.c_str() || *end || value[0] == '-') {
                    error = "invalid sampling seed";
                    return false;
                }
                out.seed = uint64_t(parsed);
            }
        }
    }
    if ((image && out.embeddings_path.empty()) || !parse_ids(id_text, out.ids, image)) {
        error = "bad prompt IDs";
        return false;
    }
    return true;
}

bool evaluate_image_prompt(Runtime& runtime, const Request& request, std::string& error) {
    std::ifstream file(request.embeddings_path, std::ios::binary);
    if (!file) { error = "cannot open image embeddings file"; return false; }
    const auto& ids = request.ids;
    const int32_t width = llama_model_n_embd_inp(runtime.model());
    if (width <= 0) { error = "model has no image input width"; return false; }
    size_t at = 0;
    size_t image_count = 0;
    while (at < ids.size()) {
        size_t end = at + 1;
        const bool image = ids[at] == -1;
        while (end < ids.size() && (ids[end] == -1) == image) ++end;
        const size_t count = end - at;
        if (image) {
            // Adjacent pictures have adjacent placeholders but remain separate
            // non-causal blocks. The SVE records supply their boundaries.
            size_t remaining = count;
            while (remaining > 0) {
                std::array<int32_t, 5> header{};
                if (!file.read(reinterpret_cast<char*>(header.data()), sizeof(header)) ||
                    header[0] != 0x31455653 || header[1] <= 0 ||
                    size_t(header[1]) > remaining || header[2] != header[1] ||
                    header[3] != 1 || header[4] != width ||
                    size_t(header[1]) > runtime.options().n_ubatch) {
                    error = "image embedding record does not match prompt";
                    return false;
                }
                const size_t rows_count = size_t(header[1]);
                std::vector<float> rows(rows_count * size_t(width));
                if (!file.read(reinterpret_cast<char*>(rows.data()),
                               std::streamsize(rows.size() * sizeof(float)))) {
                    error = "short image embedding record";
                    return false;
                }
                if (!runtime.decode_image(rows.data(), rows_count, runtime.position(),
                                          end == ids.size() && rows_count == remaining, error)) return false;
                remaining -= rows_count;
                ++image_count;
            }
        } else {
            for (size_t offset = at; offset < end;) {
                const size_t n = std::min<size_t>(runtime.options().n_batch, end - offset);
                if (runtime.decode_tokens(ids.data() + offset, n, runtime.position(),
                                          offset + n == ids.size(), error) != 0) return false;
                offset += n;
            }
        }
        at = end;
    }
    if (image_count == 0 || file.peek() != std::char_traits<char>::eof()) {
        error = "image embedding count does not match prompt";
        return false;
    }
    return true;
}

int generate(Runtime& runtime, const Request& request) {
    if (request.ids.size() + size_t(request.limit) > runtime.options().n_ctx) {
        std::puts("ERR request exceeds model context");
        return 0;
    }
    std::string error;
    llama_synchronize(runtime.context());
    runtime.reset(true);
    const auto start = Clock::now();
    const bool evaluated = request.embeddings_path.empty()
        ? runtime.evaluate(request.ids, error)
        : evaluate_image_prompt(runtime, request, error);
    if (!evaluated) {
        std::printf("ERR %s\n", error.c_str());
        return 0;
    }
    const double pp_ms = milliseconds(start);
    std::printf("PP %zu %zu %.3f %.3f\n", request.ids.size(), request.ids.size(),
                pp_ms, request.ids.size() * 1000.0 / std::max(0.001, pp_ms));
    const auto tg_start = Clock::now();
    int emitted = 0;
    int offered = 0;
    int accepted = 0;
    bool stopped = false;
    auto send = [&](llama_token token) {
        std::printf("T %d\n", token);
        ++emitted;
        stopped = !request.ignore_eog && runtime.is_eog(token);
    };
    // Greedy MTP verification cannot preserve a stochastic target distribution.
    const bool use_mtp = request.temperature == 0.0 && runtime.mtp_enabled() && request.embeddings_path.empty();
    std::mt19937_64 rng(request.seed ? request.seed : std::random_device{}());
    auto choose = [&]() { return request.temperature > 0.0 ? sample_logits(runtime, request, rng)
                                                       : runtime.greedy(request.ignore_eog); };
    if (use_mtp && request.limit > 1 && request.ids.size() >= 2) runtime.mtp_begin(request.ids);
    llama_token pending = choose();
    send(pending);
    while (!stopped && emitted < request.limit) {
        if (!use_mtp || request.ids.size() < 2 || request.limit - emitted == 1) {
            if (runtime.decode_tokens(&pending, 1, runtime.position(), true, error)) {
                std::printf("ERR %s\n", error.c_str());
                return 0;
            }
            pending = choose();
            send(pending);
            continue;
        }
        const llama_pos position = runtime.position();
        const llama_token draft = runtime.mtp_draft(pending);
        ++offered;
        const llama_token verify[] = {pending, draft};
        if (runtime.decode_tokens(verify, 2, position, false, error, true)) {
            std::printf("ERR %s\n", error.c_str());
            return 0;
        }
        const llama_token target = runtime.greedy_at(0);
        if (draft != target) {
            runtime.rollback(position + 1);
            pending = target;
            send(target);
            continue;
        }
        ++accepted;
        send(draft);
        if (stopped || emitted >= request.limit) break;
        const llama_token bonus = runtime.greedy_at(1);
        runtime.mtp_accept_catchup(draft);
        pending = bonus;
        send(bonus);
    }
    std::printf("DONE %d %zu %.3f %.3f %s %d %d 0\n", emitted, request.ids.size(), pp_ms,
                milliseconds(tg_start), stopped ? "stop" : "limit", accepted, offered);
    return 0;
}
}  // namespace

int strata_deepseek_v4_serve(int argc, char** argv) {
    try {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        strata::deepseek_v4::Options options;
        std::vector<std::string> remaining;
        std::string error;
        if (!strata::deepseek_v4::parse_options(argc, argv, options, remaining, error)) {
            std::fprintf(stderr, "strata deepseek_v4: %s\n", error.c_str());
            return 2;
        }
        bool serve = false;
        bool vision = false;
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (remaining[i] == "--serve") serve = true;
            else if (remaining[i] == "--vision") vision = true;
            else if (remaining[i] == "--family" && i + 1 < remaining.size() && remaining[++i] == "deepseek_v4") {}
            else if (remaining[i] == "--help") {
                std::fputs(strata::deepseek_v4::options_usage(), stdout);
                return 0;
            } else {
                std::fprintf(stderr, "strata deepseek_v4: unsupported option %s\n", remaining[i].c_str());
                return 2;
            }
        }
        if (!serve) {
            std::fputs("strata deepseek_v4: --serve is required\n", stderr);
            return 2;
        }
        if (!strata::deepseek_v4::validate_options(options, error)) {
            std::fprintf(stderr, "strata deepseek_v4: %s\n", error.c_str());
            return 2;
        }
        if (vision && (!Runtime::image_api_available() || options.mmproj.empty())) {
            std::fputs("strata deepseek_v4: --vision requires --mmproj and the image API\n", stderr);
            return 2;
        }
        auto runtime = Runtime::load(options, error);
        if (!runtime) {
            std::fprintf(stderr, "strata deepseek_v4: %s\n", error.c_str());
            return 1;
        }
        std::printf("INFO family=deepseek_v4 mtp=%d\nREADY %u\n", runtime->mtp_enabled() ? 1 : 0,
                    options.n_ctx);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT" || line == "EXIT") break;
            try {
                if (line.rfind("TOKENIZE ", 0) == 0) {
                    const auto tokens = runtime->tokenize(line.substr(9), false, true);
                    std::fputs("IDS", stdout);
                    for (auto token : tokens) std::printf(" %d", token);
                    std::putchar('\n');
                    continue;
                }
                Request request;
                if (!parse_request(line, request, error)) {
                    std::printf("ERR %s\n", error.c_str());
                    continue;
                }
                if (!vision && !request.embeddings_path.empty()) {
                    std::puts("ERR this engine was started without --vision");
                    continue;
                }
                generate(*runtime, request);
            } catch (const std::exception& failure) {
                std::printf("ERR %s\n", failure.what());
            }
        }
        return 0;
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "strata deepseek_v4 fatal: %s\n", failure.what());
        return 1;
    }
}
