#include "strata/gemma4/service.hpp"
#include "strata/gemma4/native/draft_round.hpp"
#include "strata/gemma4/native/pp_hh_only349_policy.hpp"
#include "strata/gemma4/native/global_gqa371_policy.hpp"
#include "attention_swa.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace g = strata::gemma4::native;
namespace n = strata::qwen35::native;
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int integer(const std::string& text) {
    size_t used = 0; const int value = std::stoi(text, &used);
    if (used != text.size()) throw std::invalid_argument("invalid integer");
    return value;
}
std::vector<int> ids(const std::string& csv, int vocabulary, bool empty = false) {
    if (csv.empty()) {
        if (empty) return {};
        throw std::invalid_argument("empty token list");
    }
    if (csv.back() == ',') throw std::invalid_argument("trailing token separator");
    std::istringstream in(csv); std::string part; std::vector<int> out;
    while (std::getline(in, part, ',')) {
        const int id = integer(part);
        if (id < 0 || id >= vocabulary) throw std::invalid_argument("token outside vocabulary");
        out.push_back(id);
    }
    return out;
}
struct Options {
    std::string model, assistant, eog = "1,50,106";
    int context = 262144, batch = 8192, mtp = 1;
    strata::kernels::SamplerParams sampling;
    bool hh = true, gqa = true, head = true, full_k = false, swa = false, history = true;
    static Options parse(int argc, char** argv) {
        std::map<std::string, std::string> values;
        for (int i = 1; i < argc; i += 2)
            if (i + 1 == argc || !values.emplace(argv[i], argv[i + 1]).second)
                throw std::invalid_argument("options require unique --key value pairs");
        auto take = [&](const char* key, std::string fallback = "") {
            const auto at = values.find(key); if (at == values.end()) return fallback;
            std::string value = at->second; values.erase(at); return value;
        };
        auto flag = [&](const char* key, bool fallback = false) {
            const int value = integer(take(key, fallback ? "1" : "0"));
            if (value != 0 && value != 1) throw std::invalid_argument(key);
            return value != 0;
        };
        Options o; o.model = take("--model"); o.assistant = take("--assistant");
        // --aux is accepted for the common native-service launcher.
        const auto aux = take("--aux");
        if (o.assistant.empty()) o.assistant = aux;
        else if (!aux.empty()) throw std::invalid_argument("duplicate assistant path");
        o.context = integer(take("--context", "262144"));
        o.batch = integer(take("--batch", "8192")); o.mtp = integer(take("--mtp", "1"));
        o.eog = take("--eog", o.eog);
        auto decimal = [&](const char* key, float fallback) {
            const auto text = take(key, std::to_string(fallback));
            size_t used = 0; const auto value = std::stof(text, &used);
            if (used != text.size()) throw std::invalid_argument(key);
            return value;
        };
        o.sampling.temperature = decimal("--temperature", 1.0f);
        o.sampling.top_k = integer(take("--top-k", "20"));
        o.sampling.top_p = decimal("--top-p", 0.95f);
        o.sampling.min_p = decimal("--min-p", 0.0f);
        const auto seed = take("--seed", "0"); size_t used = 0;
        o.sampling.seed = std::stoull(seed, &used);
        if (used != seed.size() || seed.empty() || seed[0] == '-') throw std::invalid_argument("seed");
        strata::kernels::RadixSampler::validate(o.sampling);
        o.hh = flag("--pp-hh349", true); o.gqa = flag("--tg-gqa371", true);
        o.head = flag("--draft-head399", true); o.full_k = flag("--ffn-full-k");
        o.swa = flag("--swa-tc403"); o.history = flag("--history-experimental", true);
        if (!values.empty() || o.model.empty() ||
            o.context != 262144 || o.batch != 8192 || o.mtp < 0 || o.mtp > 1 ||
            (o.mtp == 0 && (!o.assistant.empty())))
            throw std::invalid_argument("Gemma requires its 256K/Q8/8192 Dense profile and MTP0 or1");
        return o;
    }
};
struct Stream {
    cudaStream_t value = nullptr;
    void open() {
        n::check_cuda(cudaSetDeviceFlags(cudaDeviceMapHost), "Gemma mapped history device flags");
        n::check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "Gemma service stream");
    }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};

class Service {
    Options options_;
    strata::GgufFile file_;
    Stream stream_;
    std::unique_ptr<g::Model> model_;
    std::unique_ptr<strata::GgufFile> assistant_file_;
    std::unique_ptr<g::assistant::Draft> draft_;
    std::unique_ptr<g::DraftRoundController> controller_;
    std::vector<int> eog_;
    strata::models::RequestSampling sampling_;
    bool request_mtp_ = false;
    int pending_ = -1, remaining_ = 0, capacity_ = 0;
    bool active_ = false, stopped_ = false, round_bound_ = false;
    bool selected_hh_ = false, selected_gqa_ = false, selected_head_ = false;
    bool is_eog(int token) const {
        return std::find(eog_.begin(), eog_.end(), token) != eog_.end();
    }
    void configure(size_t prompt, int capacity, int maximum_new) {
        const bool exact = (prompt == 8192 && capacity == 10240) ||
                           (prompt == 30720 && capacity == 32768);
        if ((options_.full_k || options_.swa) && (!exact || maximum_new != 2048))
            throw std::invalid_argument("fullK/SWA options require original8K/30K+2048 shapes");
        // Preserve the established scoped combination without applying it to
        // untested shapes or changing the >32K mapped-history algorithm.
        selected_hh_ = options_.hh && exact && maximum_new == 2048;
        selected_gqa_ = options_.gqa && request_mtp_ && exact && maximum_new == 2048;
        selected_head_ = options_.head && request_mtp_ && exact && maximum_new == 2048;
        if (controller_) controller_->release_graphs();
        model_->reset(capacity); model_->set_profile(nullptr);
        // The retained305/319 baseline, expressed as named APIs rather than a
        // benchmark bitfield or a dependency on historical driver headers.
        model_->select_library_pp(true); model_->select_vector3(true);
        model_->select_grouped(true); model_->select_full_down(true);
        model_->select_paired_pp(false); model_->select_native_pp(false);
        model_->select_pp_algorithm(-1); model_->select_main_slice(options_.full_k ? 0 : 896);
        model_->select_all_pp(true); model_->select_attention(true, true);
        model_->select_half_pp(true); model_->select_half_attention(false);
        model_->select_single_ffn(true); model_->select_row_half(false);
        model_->select_fast_ffn(true); model_->select_fused_attention(true);
        model_->select_periodic_row(false); model_->select_compact_half(true);
        model_->select_projection_q8_mask(32748); model_->select_t1_q4_ffn(g::T1Q4FfnPolicy::Both);
        model_->select_t1_paths(636); model_->select_aligned_q6_head(false);
        model_->select_three_panel_half(prompt == 1280);
        model_->select_short_pp_backend(prompt == 1280 ? g::ShortPpBackend::FastFullK : g::ShortPpBackend::ExistingPanel3);
        model_->select_global_shared_long(false);
        model_->select_physical_global_pp(true); model_->select_compensated_physical_global_pp(true);
        model_->select_pp_three_product333(false);
        model_->select_pp_key_tile(selected_hh_ ? g::pp349::Keys1024HH349 : g::PpKeyTile::Keys512);
        model_->select_global_verify_merge(selected_gqa_ ? g::tg371::Gqa371 : g::GlobalVerifyMerge::Parallel321);
        if (draft_) draft_->select_head_backend(selected_head_ ? g::assistant::HeadBackend::Direct399 :
                                                               g::assistant::HeadBackend::DecodeSgemm);
    }
public:
    explicit Service(Options options) : options_(std::move(options)), file_(options_.model) {
        stream_.open();
        model_ = std::make_unique<g::Model>(file_, options_.batch, stream_.value, g::Q6DownStorage::Aligned224, true);
        eog_ = ids(options_.eog, model_->spec().vocabulary, true);
        if (eog_.size() > 16) throw std::invalid_argument("at most16 EOG IDs");
        sampling_.params = options_.sampling; sampling_.configured = true;
    }
    void prepare_assistant() {
        if (!controller_ && options_.mtp && !options_.assistant.empty()) {
            const auto scratch = model_->draft_scratch();
            assistant_file_ = std::make_unique<strata::GgufFile>(options_.assistant);
            draft_ = std::make_unique<g::assistant::Draft>(*assistant_file_, stream_.value, scratch.decoded, scratch.decoded_bytes);
            draft_->select_split_attention(true);
            controller_ = std::make_unique<g::DraftRoundController>(*model_, *draft_);
        }
    }
    // Reverse member destruction releases Controller graphs, then Draft, Model
    // and finally stream; the immutable GGUF mappings outlive their consumers.
    ~Service() = default;
    void release() {
        if (controller_) controller_->release_graphs();
        model_->reset(1); active_ = stopped_ = round_bound_ = false;
        pending_ = -1; remaining_ = capacity_ = 0;
    }
    void command(const std::string& line) {
        std::istringstream in(line); std::string command, extra; int slot = -1;
        if (!(in >> command >> slot) || slot != 0) throw std::invalid_argument("Gemma requires slot0");
        if (command == "CONFIG") {
            if (active_) throw std::invalid_argument("sampling configuration requires an idle request");
            sampling_.configure(in, options_.context);
            std::cout << "CONFIGURED slot=0\n"; return;
        }
        if (command == "RELEASE") {
            if (in >> extra) throw std::invalid_argument("unexpected RELEASE field");
            release(); std::cout << "RELEASED slot=0\n"; return;
        }
        if (command == "INFO") {
            if (in >> extra) throw std::invalid_argument("unexpected INFO field");
            const auto stats = controller_ && round_bound_ ? controller_->stats() : g::DraftRoundStats{};
            const bool graph = stats.captures[0] || stats.captures[1] || stats.captures[2] || model_->graph_replays();
            std::cout << "INFO requested_mtp=" << options_.mtp << " actual_mtp=" << (round_bound_ ? options_.mtp : 0)
                << " sampling=" << sampling_.enabled()
                << " mtp_sampling_fallback=" << (sampling_.enabled() && options_.mtp)
                << " cuda_graph=" << graph << " active=" << active_ << " context=" << options_.context
                << " logical_request_capacity=" << capacity_ << " global_kv_window=32768"
                << " kv_bits=8 position=" << model_->position()
                << " own_kv_bytes=" << model_->kv_bytes() << " own_history_ram_bytes=" << model_->history_ram_bytes()
                << " own_history_resident_bytes=" << model_->history_resident_bytes()
                << " Draft_metadata_context=" << (draft_ ? draft_->metadata_context() : 0)
                << " Draft_runtime_context=" << (draft_ ? draft_->runtime_context() : 0)
                << " rounds=" << stats.rounds << " accepted=" << stats.accepted
                << " base_T1_tail_rounds=" << stats.base_T1_tail_rounds
                << " tail_assistant_context=" << stats.tail_assistant_context
                << " pp_hh349=" << selected_hh_ << " tg_gqa371=" << selected_gqa_
                << " draft_head399=" << selected_head_ << " ffn_full_k=" << options_.full_k
                << " swa_tc403=" << options_.swa << " history_experimental=" << options_.history
                << " history_selection_quality_scope=separate_finite_review"
                << " resource_occupancy_measured=0 model_quality_admitted=0\n"; return;
        }
        if (command == "PREPARE") {
            int maximum_new = 0; std::string csv;
            if (!(in >> maximum_new >> csv) || in >> extra || active_ || maximum_new < 1)
                throw std::invalid_argument("invalid Gemma PREPARE");
            const auto prompt = ids(csv, model_->spec().vocabulary);
            if (prompt.size() > size_t(options_.context) || size_t(maximum_new) > size_t(options_.context) - prompt.size())
                throw std::invalid_argument("request exceeds context");
            capacity_ = int(prompt.size()) + maximum_new;
            if (capacity_ > 32768 && !options_.history)
                throw std::invalid_argument("history beyond32K requires explicit experimental option; no tokens discarded");
            if (!sampling_.enabled()) prepare_assistant();
            request_mtp_ = controller_ && !sampling_.enabled();
            configure(prompt.size(), capacity_, maximum_new);
            model_->configure_sampling(sampling_.params);
            const auto start = Clock::now();
            {
                const auto lease = model_->draft_cache(59);
                g::swa403::RequestScope scope({lease.owner, lease.epoch, stream_.value}, model_->position(), options_.swa);
                pending_ = model_->prefill(prompt);
            } // All PP work joined by prefill; the host-only scope ends before TG Graph capture.
            remaining_ = maximum_new - 1; stopped_ = !remaining_ || is_eog(pending_); active_ = true;
            round_bound_ = request_mtp_ && !stopped_;
            if (round_bound_) controller_->bind(remaining_, eog_, options_.mtp, capacity_);
            std::cout << "PREFILL anchor=" << pending_ << " tokens=" << prompt.size()
                << " ms=" << milliseconds(start) << '\n'; return;
        }
        if (command == "STEP") {
            int anchor = -1, requested = 0;
            if (!(in >> anchor >> requested) || in >> extra || !active_ || stopped_ ||
                anchor != pending_ || requested < 1 || requested > remaining_ ||
                (round_bound_ && requested < (std::min)(2, remaining_)))
                throw std::invalid_argument("invalid Gemma STEP or pending/budget boundary");
            const auto start = Clock::now(); std::vector<int> output;
            if (round_bound_) {
                const auto response = controller_->round();
                if (response.count < 1 || response.count > requested || response.count > remaining_)
                    throw std::runtime_error("Gemma round output extent");
                output.assign(response.ids, response.ids + response.count);
                stopped_ = response.reason != g::VerifyStop::Continue;
            } else output.push_back(model_->step(true));
            remaining_ -= int(output.size()); pending_ = output.back();
            stopped_ = stopped_ || !remaining_ || is_eog(pending_);
            std::cout << "STEPPED next=" << pending_ << " ms=" << milliseconds(start) << " ids=";
            for (size_t i = 0; i < output.size(); ++i) std::cout << (i ? "," : "") << output[i];
            std::cout << '\n'; return;
        }
        throw std::invalid_argument("unsupported Gemma command");
    }
};
}

int strata_gemma4_native_serve(int argc, char** argv) {
    try {
        const auto options = Options::parse(argc, argv); Service service(options);
        std::cout << "READY slots=1 family=gemma4_dense resource_occupancy_measured=0\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            try {
                if (line.size() > size_t(options.context) * 24 + 1024) throw std::invalid_argument("Gemma command too long");
                service.command(line);
            } catch (const std::exception& error) {
                service.release(); std::cout << "ERROR " << error.what() << '\n';
            }
            std::cout << std::flush;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Gemma service: " << error.what() << '\n'; return 1;
    }
}
