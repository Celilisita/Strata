#include "runtime.hpp"
#include "options.hpp"
#include "device_cache.hpp"
#include "serve_input.hpp"
#include "serve_request.hpp"
#include "serve_finish.hpp"
#include "runtime_memory.hpp"
#include "serve_metrics.hpp"
#include "../glm5_next/kernel_activity.hpp"
#include "../glm5_next/serve_mtp.hpp"

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

using Request = strata::deepseek_v4::ServeRequest;

llama_token sample_logits(Runtime& runtime,const Request& request,std::mt19937_64& rng,
                          const strata::deepseek_v4::ServeHistory& history){
    const auto token=strata::deepseek_v4::serve_sample(llama_get_logits_ith(runtime.context(),-1),
        llama_vocab_n_tokens(runtime.vocab()),request,history,rng,[&](llama_token id){return runtime.is_eog(id);});
    strata::deepseek_v4::idle_device_expert_cache(runtime.context());
    return token;
}

bool evaluate_image_prompt(Runtime& runtime, const Request& request, std::string& error,const std::atomic<bool>* cancel) {
    std::ifstream file(request.embeddings_path, std::ios::binary);
    if (!file) { error = "cannot open image embeddings file"; return false; }
    const auto& ids = request.ids;
    const int32_t width = llama_model_n_embd_inp(runtime.model());
    if (width <= 0) { error = "model has no image input width"; return false; }
    size_t at = 0;
    size_t image_count = 0;
    const bool glm = runtime.metadata("general.architecture") == "glm5-next";
    while (at < ids.size()) {
        if(cancel&&cancel->load()){error="request cancelled";return false;}
        size_t end = at + 1;
        const bool image = ids[at] == -1;
        while (end < ids.size() && (ids[end] == -1) == image) ++end;
        const size_t count = end - at;
        if (image) {
            // Adjacent pictures have adjacent placeholders but remain separate
            // non-causal blocks. The SVE records supply their boundaries.
            size_t remaining = count;
            while (remaining > 0) {
                if(cancel&&cancel->load()){error="request cancelled";return false;}
                std::array<int32_t, 5> header{};
                if (!file.read(reinterpret_cast<char*>(header.data()), sizeof(header)) ||
                    header[0] != 0x31455653 || header[1] <= 0 ||
                    size_t(header[1]) > remaining || header[2] != header[1] ||
                    header[3] != 1 || header[4] != width ||
                    (!glm && size_t(header[1]) > runtime.options().n_ubatch)) {
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
                for(size_t offset=0;offset<rows_count;) {
                    const size_t n=std::min<size_t>(runtime.options().n_ubatch, rows_count-offset);
                    if (!runtime.decode_image(rows.data()+offset*size_t(width), n, runtime.position(),
                        end==ids.size() && rows_count==remaining && offset+n==rows_count,error))return false;
                    offset+=n;
                }
                remaining -= rows_count;
                ++image_count;
            }
        } else {
            for (size_t offset = at; offset < end;) {
                if(cancel&&cancel->load()){error="request cancelled";return false;}
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

int generate(Runtime& runtime, const Request& request,const std::shared_ptr<std::atomic<bool>>& cancel) {
    if (request.ids.size() + size_t(request.limit) > runtime.options().n_ctx) {
        std::puts("ERR request exceeds model context");
        return 0;
    }
    std::string error;
    for(auto id:request.ids)if(id>=llama_vocab_n_tokens(runtime.vocab())) {
        std::puts("ERR prompt token ID lies outside the model vocabulary");return 0;
    }
    bool request_mtp=false;
    if(!strata::deepseek_v4::resolve_serve_mtp(request,runtime.glm_mtp_enabled()||runtime.mtp_enabled(),request_mtp,error,
                                            runtime.glm_mtp_enabled())) {
        std::printf("ERR %s\n",error.c_str());return 0;
    }
    llama_synchronize(runtime.context());
    strata::deepseek_v4::settle_device_background(runtime.context());
    runtime.reset(true);
    const bool glm_mtp=runtime.glm_mtp_enabled()&&request_mtp;
    runtime.glm_mtp_request(glm_mtp);
    std::fprintf(stderr,"strata_serve_request effective_mtp=%d mtp_requested=%d temperature=%.17g top_k=%d top_p=%.17g min_p=%.17g seed_given=%d seed=%llu penalty_repeat=%.17g penalty_freq=%.17g penalty_present=%.17g penalty_last_n=%d max_new=%d image=%d ignore_eog=%d\n",
        int(request_mtp&&(runtime.glm_mtp_enabled()||request.ids.size()>=2)),request.mtp,
        request.temperature,request.top_k,request.top_p,request.min_p,int(request.seed_given),
        (unsigned long long)request.seed,request.penalty_repeat,request.penalty_freq,
        request.penalty_present,request.penalty_last_n,request.limit,int(!request.embeddings_path.empty()),int(request.ignore_eog));
    std::fflush(stderr);
    const strata::deepseek_v4::ServeMetrics counters(runtime);
    strata::glm5_next::KernelActivity kernel_activity(strata::glm5_next::kernel_activity_requested());
    const auto start = Clock::now();
    const bool evaluated = request.embeddings_path.empty()
        ? runtime.evaluate(request.ids, error,cancel.get())
        : evaluate_image_prompt(runtime, request, error,cancel.get());
    if(!evaluated&&cancel->load()){
        llama_synchronize(runtime.context());
        strata::deepseek_v4::settle_device_background(runtime.context());
        const double pp_ms=milliseconds(start);
        strata::deepseek_v4::report_serve_finish("cancel",0,request.ids.size(),size_t(runtime.position()),pp_ms,0);
        std::printf("DONE 0 %zu %.3f 0 cancel 0 0 0\n",request.ids.size(),pp_ms);return 0;
    }
    if (!evaluated) throw std::runtime_error(error);
    // Complete the request boundary before reporting PP, as the first sample requires.
    llama_synchronize(runtime.context());
    if(glm_mtp)
        return strata::glm5_next::serve_mtp(runtime,request,cancel,start,counters,kernel_activity);
    const double pp_ms = milliseconds(start);
    const auto pp_activity=kernel_activity.snapshot();
    strata::deepseek_v4::runtime_memory("after-prefill");
    std::printf("PP %zu %zu %.3f %.3f\n", request.ids.size(), request.ids.size(),
                pp_ms, request.ids.size() * 1000.0 / std::max(0.001, pp_ms));
    const auto tg_start = Clock::now();
    int emitted = 0;
    int offered = 0;
    int accepted = 0;
    bool stopped = false;
    strata::deepseek_v4::ServeHistory history(request,request.ids);
    auto send = [&](llama_token token) {
        if(cancel->load())return;
        history.remember(token);
        std::printf("T %d\n", token);
        ++emitted;
        stopped = !request.ignore_eog && runtime.is_eog(token);
    };
    // Greedy MTP verification cannot preserve a stochastic target distribution.
    const bool use_mtp = request_mtp && runtime.mtp_enabled();
    auto rng = strata::deepseek_v4::serve_rng(request);
    auto choose = [&]() { return (request.temperature > 0.0 || request.penalties()) ? sample_logits(runtime, request, rng,history)
                                                       : runtime.greedy(request.ignore_eog); };
    if (use_mtp && request.limit > 1 && request.ids.size() >= 2) runtime.mtp_begin(request.ids);
    llama_token pending = cancel->load() ? 0 : choose();
    send(pending);
    while (!stopped && emitted < request.limit && !cancel->load()) {
        if (!use_mtp || request.ids.size() < 2 || request.limit - emitted == 1) {
            if (runtime.decode_tokens(&pending, 1, runtime.position(), true, error)) {
                throw std::runtime_error(error);
            }
            if(cancel->load())break;
            pending = choose();
            send(pending);
            continue;
        }
        const llama_pos position = runtime.position();
        const llama_token draft = runtime.mtp_draft(pending);
        ++offered;
        const llama_token verify[] = {pending, draft};
        if (runtime.decode_tokens(verify, 2, position, false, error, true)) {
            throw std::runtime_error(error);
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
    const char* finish=strata::deepseek_v4::serve_finish_reason(stopped,size_t(emitted),size_t(request.limit));
    llama_synchronize(runtime.context());
    strata::deepseek_v4::settle_device_background(runtime.context());
    const auto tg_ms=milliseconds(tg_start);counters.report(runtime);
    if(kernel_activity.active()){
        const auto counts=kernel_activity.finish();
        std::fprintf(stderr,"strata_kernel_activity pp_kernels=%llu tg_kernels=%llu dropped=%llu errors=%llu profiled=1\n",
            (unsigned long long)pp_activity.kernels,(unsigned long long)(counts.kernels-pp_activity.kernels),
            (unsigned long long)counts.dropped,(unsigned long long)counts.errors);
        std::fflush(stderr);
        if(counts.dropped||counts.errors||!counts.kernels)throw std::runtime_error("kernel activity incomplete");
    }
    strata::deepseek_v4::report_serve_finish(finish,size_t(emitted),request.ids.size(),request.ids.size(),pp_ms,tg_ms);
    std::printf("DONE %d %zu %.3f %.3f %s %d %d 0\n", emitted, request.ids.size(), pp_ms,
                tg_ms, finish, accepted, offered);
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
            else if (remaining[i] == "--family" && i + 1 < remaining.size() && (remaining[++i] == "deepseek_v4" || remaining[i] == "glm5_next")) {}
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
        std::printf("INFO family=%s mtp=%d\nREADY %u stop\n",
                    runtime->metadata("general.architecture") == "glm5-next" ? "glm5_next" : "deepseek_v4", (runtime->mtp_enabled()||runtime->glm_mtp_enabled()) ? 1 : 0,
                    options.n_ctx);
        strata::deepseek_v4::ServeInput input;
        std::string line;std::shared_ptr<std::atomic<bool>> cancel;
        while (input.next(line,cancel)) {
            strata::deepseek_v4::ServeInputLease lease{input,cancel};
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
                if (!strata::deepseek_v4::parse_serve_request(line, request, error)) {
                    std::printf("ERR %s\n", error.c_str());
                    continue;
                }
                if (!vision && !request.embeddings_path.empty()) {
                    std::puts("ERR this engine was started without --vision");
                    continue;
                }
                generate(*runtime, request,cancel);
            } catch (const std::invalid_argument& failure) {
                // A rejected sampling policy can reset cleanly without
                // treating it as a model/provider execution failure.
                llama_synchronize(runtime->context());
                strata::deepseek_v4::settle_device_background(runtime->context());
                runtime->reset(true);
                std::printf("ERR %s\n",failure.what());
            } catch (const std::exception& failure) {
                std::printf("ERR %s\n", failure.what());
                // A GLM provider/state failure is terminal for this context.
                // Drain and release it; the public frontend may reload on the
                // next request rather than reusing a poisoned slot directory.
                if(runtime->metadata("general.architecture")=="glm5-next")throw;
            }
        }
        return 0;
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "strata deepseek_v4 fatal: %s\n", failure.what());
        return 1;
    }
}
