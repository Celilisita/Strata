#include "strata/models/native_options.hpp"
#include "strata/models/request_sampling.hpp"
#include "strata/qwen35/native/mtp.hpp"
#include "service_profile.hpp"
#include "service_decode_graph.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

namespace {
using namespace strata::qwen35::native;
using Clock = std::chrono::steady_clock;
struct Stream {
    cudaStream_t p = nullptr;
    void open() { check_cuda(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking), "Dense service stream"); }
    ~Stream() { if (p) { cudaStreamSynchronize(p); cudaStreamDestroy(p); } }
};
std::vector<int> parse_ids(const std::string& csv, int vocab) {
    if (csv.empty() || csv.back() == ',') throw std::invalid_argument("empty token list");
    std::istringstream input(csv); std::string piece; std::vector<int> ids;
    while (std::getline(input, piece, ',')) {
        size_t used = 0; int id = std::stoi(piece, &used);
        if (used != piece.size() || id < 0 || id >= vocab) throw std::invalid_argument("token outside vocabulary");
        ids.push_back(id);
    }
    return ids;
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
class DenseService {
    const strata::models::NativeOptions options_;
    strata::GgufFile file_;
    strata::qwen35::Spec spec_;
    Stream stream_;
    std::unique_ptr<DenseWeights> weights_;
    std::unique_ptr<DenseGemm> fallback_, gemm_;
    std::unique_ptr<FfnShapePolicy> shape_;
    std::unique_ptr<DenseModel> model_;
    DeviceStorage storage_;
    int* input_ = nullptr;
    int* history_ = nullptr;
    std::unique_ptr<strata::kernels::RadixSampler> sampler_;
    strata::models::RequestSampling sampling_;
    std::unique_ptr<NativeMtp> mtp_;
    std::unique_ptr<ServiceDecodeGraph> graph_;
    int position_ = -1, pending_ = -1, remaining_ = 0;
    bool active_ = false, speculative_ = false;
    int pick() {
        int result = -1; unsigned invalid = 0;
        check_cuda(cudaMemcpyAsync(&invalid, model_->invalid_logits(), 4, cudaMemcpyDeviceToHost, stream_.p), "Dense finite flag");
        if (sampling_.enabled()) {
            if (!sampler_) sampler_ = std::make_unique<strata::kernels::RadixSampler>(int(spec_.n_vocab));
            const size_t bytes = sampling_.history.size() * sizeof(int);
            if (bytes) check_cuda(cudaMemcpyAsync(history_, sampling_.history.data(), bytes, cudaMemcpyHostToDevice, stream_.p), "Dense penalty history");
            sampler_->sample(model_->logits(), history_, int(sampling_.history.size()), sampling_.params, stream_.p);
            strata::kernels::SamplePick sample;
            check_cuda(cudaMemcpyAsync(&sample, sampler_->result(), sizeof(sample), cudaMemcpyDeviceToHost, stream_.p), "Dense sampled response");
            check_cuda(cudaStreamSynchronize(stream_.p), "Dense sampling completion");
            if (sample.first_bad != -1) throw std::runtime_error("Dense nonfinite logits");
            result = sample.id; ++sampling_.params.counter;
            // The device-picked state input, not just the HTTP response, changes.
            check_cuda(cudaMemcpyAsync(const_cast<int*>(model_->picked_token()), &result, 4,
                                       cudaMemcpyHostToDevice, stream_.p), "Dense sampled next-state input");
        } else check_cuda(cudaMemcpyAsync(&result, model_->picked_token(), 4, cudaMemcpyDeviceToHost, stream_.p), "Dense greedy response");
        check_cuda(cudaStreamSynchronize(stream_.p), "Dense token completion");
        if (invalid || result < 0 || result >= int(spec_.n_vocab)) throw std::runtime_error("Dense invalid logits or token");
        return result;
    }
    void release() {
        // Graphs borrow KV, prefix input and sampler state; drain them first.
        graph_.reset();
        if (mtp_) mtp_->release_graphs();
        check_cuda(cudaStreamSynchronize(stream_.p), "Dense request drain");
        model_->set_visual_input(nullptr,stream_.p);
        model_->reset_prefill(stream_.p);
        active_ = speculative_ = false; position_ = pending_ = -1;
        remaining_ = 0; sampling_ = {};
    }
public:
    explicit DenseService(strata::models::NativeOptions o) : options_(std::move(o)), file_(options_.model) {
        std::string error;
        if (!strata::qwen35::load_spec(file_, spec_, error)) throw std::runtime_error(error);
        if (options_.context != int(spec_.context_length))
            throw std::invalid_argument("Dense native profile requires the model context length");
        stream_.open();
        weights_ = std::make_unique<DenseWeights>(file_, spec_, stream_.p);
        fallback_ = make_cublas_pair_gemm(stream_.p);
        gemm_ = make_cutlass_swiglu_gemm(*fallback_, stream_.p);
        shape_ = std::make_unique<FfnShapePolicy>(*gemm_, 8192);
        model_ = std::make_unique<DenseModel>(spec_, *weights_, service_limits(options_.context, options_.kv == "ram8k-reuse"), stream_.p);
        input_ = static_cast<int*>(storage_.allocate(size_t(options_.context) * 4));
        history_ = static_cast<int*>(storage_.allocate(size_t(options_.context) * 4));
        check_cuda(cudaStreamSynchronize(stream_.p), "Dense service ready");
    }
    void command(const std::string& line) {
        std::istringstream in(line); std::string command, csv, extra; int slot;
        if (!(in >> command >> slot) || slot != 0) throw std::invalid_argument("Dense requires slot 0");
        if (command == "RELEASE") {
            if (in >> extra) throw std::invalid_argument("unexpected RELEASE field");
            release(); std::cout << "RELEASED slot=0\n"; return;
        }
        if (command == "CONFIG") {
            if (active_) throw std::invalid_argument("Dense slot is active");
            sampling_.configure(in, options_.context); std::cout << "CONFIGURED slot=0\n"; return;
        }
        if (command == "PREPARE" || command == "PREPAREV") {
            int max_new;
            if (!(in >> max_new >> csv) || in >> extra || active_) throw std::invalid_argument("invalid Dense PREPARE");
            auto visual = command=="PREPAREV" ? read_visual_prompt(csv,options_.context,int(spec_.n_embd),int(spec_.n_vocab)) : nullptr;
            const auto prompt = visual ? visual->ids : parse_ids(csv, int(spec_.n_vocab));
            if (prompt.empty() || max_new < 1 || prompt.size() + size_t(max_new) > size_t(options_.context))
                throw std::invalid_argument("Dense prompt plus output exceeds context");
            graph_.reset(); if (mtp_) mtp_->release_graphs();
            model_->set_visual_input(std::move(visual),stream_.p);
            const int maximum = int(prompt.size()) + max_new;
            model_->reserve_cache(maximum, stream_.p);
            speculative_ = options_.mtp > 0 && !sampling_.enabled() && prompt.size() >= 2 && max_new >= 3;
            apply_service_profile(*model_, *shape_, int(prompt.size()), speculative_ ? options_.mtp : 0);
            model_->reset_prefill(stream_.p); sampling_.prepare(prompt);
            check_cuda(cudaMemcpyAsync(input_, prompt.data(), prompt.size() * 4, cudaMemcpyHostToDevice, stream_.p), "Dense prompt payload");
            const auto start = Clock::now();
            if (speculative_) {
                if (!mtp_) {
                    mtp_ = std::make_unique<NativeMtp>(*model_, file_, stream_.p);
                    mtp_->use_gpu_draft_chain(true); mtp_->use_prefix_states(options_.mtp == 2);
                }
                mtp_->reserve_cache(maximum, stream_.p);
                // Existing eager warmup admits shapes before graph capture. It is
                // request work, recorded in PP time, never a hidden startup test.
                const int tokens = int(prompt.size());
                const int warm = (std::min)(tokens, model_->max_tokens() + tokens % model_->max_tokens());
                mtp_->prefill(input_, warm, stream_.p);
                check_cuda(cudaStreamSynchronize(stream_.p), "Dense MTP shape preparation");
                mtp_->enable_graphs(input_, int(prompt.size()), maximum, stream_.p);
                model_->reset_prefill(stream_.p); mtp_->prefill(input_, int(prompt.size()), stream_.p);
            } else model_->prefill(input_, int(prompt.size()), stream_.p);
            pending_ = pick(); position_ = int(prompt.size()); remaining_ = max_new - 1;
            if (!speculative_ && remaining_ && position_ <= 262142)
                graph_ = std::make_unique<ServiceDecodeGraph>(*model_, position_, maximum, stream_.p);
            active_ = true;
            std::cout << "PREFILL anchor=" << pending_ << " tokens=" << prompt.size()
                      << " pp_ms=" << elapsed(start) << " aux=0 cache_slots=0\n";
            return;
        }
        if (command == "STEP") {
            int anchor, requested;
            if (!(in >> anchor >> requested) || in >> extra || !active_ || anchor != pending_ || requested < 1 ||
                requested > remaining_ || position_ >= options_.context)
                throw std::invalid_argument("invalid Dense STEP or pending anchor");
            const auto start = Clock::now(); std::vector<int> picks;
            if (speculative_) picks = mtp_->round(anchor, position_, (std::min)(options_.mtp, requested - 1), stream_.p);
            else {
                sampling_.remember(anchor);
                if (graph_) graph_->launch(position_); else model_->decode(position_, stream_.p);
                picks.push_back(pick());
            }
            if (picks.empty() || int(picks.size()) > requested) throw std::runtime_error("Dense output extent");
            position_ += int(picks.size()); remaining_ -= int(picks.size()); pending_ = picks.back();
            std::cout << "STEPPED next=" << pending_ << " ms=" << elapsed(start) << " ids=";
            for (size_t i = 0; i < picks.size(); ++i) std::cout << (i ? "," : "") << picks[i];
            std::cout << '\n'; return;
        }
        throw std::invalid_argument("unsupported Dense command");
    }
    void recover() { release(); }
};
} // namespace
int strata_qwen35_dense_serve(int argc, char** argv) {
    try {
        const auto options = strata::models::NativeOptions::parse(argc, argv);
        DenseService service(options);
        std::cout << "READY slots=1\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            try {
                if (line.size() > size_t(options.context) * 24 + 1024) throw std::invalid_argument("Dense command too long");
                service.command(line);
            } catch (const std::exception& e) {
                service.recover(); std::cout << "ERROR " << e.what() << '\n';
            }
            std::cout << std::flush;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "Dense service: " << e.what() << '\n'; return 1; }
}
