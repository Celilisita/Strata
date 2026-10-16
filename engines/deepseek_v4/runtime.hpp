#pragma once

#include "options.hpp"
#include "expert_staging.hpp"
#include "cold_expert_provider.hpp"
#include "mtp_stage_metrics.hpp"
#include "llama.h"

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace strata::deepseek_v4 {
class MtpSidecar;class MtpDrafter;
class ExpertPageWarm;

// In-process canonical DeepSeek4 math, original quantized tensors, and one GGML
// expert-cache authority shared by prefill and decode. No external server.
class Runtime {
public:
    static std::unique_ptr<Runtime> load(
        const Options& options, std::string& error,
        ggml_backend_sched_eval_callback callback = nullptr, void* callback_data = nullptr);
    ~Runtime();
    // Rebuild only context/cache, retaining model mappings and dense GPU weights.
    bool reconfigure_context(const Options&, std::string& error,
        ggml_backend_sched_eval_callback callback = nullptr, void* callback_data = nullptr);
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    std::vector<llama_token> tokenize(std::string_view text, bool add_special = true,
                                     bool parse_special = true) const;
    std::string piece(llama_token token, bool special = false) const;
    std::string detokenize(const std::vector<llama_token>& tokens, bool special = false) const;
    bool evaluate(const std::vector<llama_token>& tokens, std::string& error);
    bool decode_image(const float* rows, size_t count, llama_pos start_pos,
                      bool logits_last, std::string& error);
    int decode_tokens(const llama_token* tokens, size_t count, llama_pos start_pos,
                      bool logits_last, std::string& error,bool all_logits=false);
    llama_token greedy(bool ignore_eog = false) const;
    llama_token greedy_at(int index,bool ignore_eog=false) const;
    bool mtp_enabled() const { return bool(mtp_weights_); }
    void mtp_begin(const std::vector<llama_token>& input);
    llama_token mtp_draft(llama_token next);
    MtpConfidence mtp_confidence() const;
    MtpStageMetrics mtp_stage_metrics() const;
    uint64_t cpu_worker_time_100ns() const;
    void mtp_set_async_input(bool enabled);
    void mtp_accept_catchup(llama_token accepted_draft);
    void mtp_single_advance();
    void rollback(llama_pos position);
    bool is_eog(llama_token token) const noexcept;
    void reset(bool clear_data = true);
    llama_perf_context_data performance() const;
    std::string metadata(const char* key) const;
    static const char* source_commit() noexcept;

    llama_model* model() const noexcept { return model_; }
    llama_context* context() const noexcept { return context_; }
    const llama_vocab* vocab() const noexcept { return vocab_; }
    ggml_backend_dev_t device() const noexcept { return devices_[0]; }
    llama_pos position() const noexcept { return position_; }
    const Options& options() const noexcept { return options_; }
    bool threadpool_resident() const noexcept { return threadpool_ != nullptr; }
    ExpertStagingMetrics expert_transport_metrics() const;
    SourcePinMetrics source_pin_metrics() const;
    ColdExpertMetrics cold_metrics() const;

    // mtmd's public C API can operate directly on model()/context()/device().
    // Linking mtmd is enabled by default; image ownership belongs to its driver.
    static bool image_api_available() noexcept;

private:
    explicit Runtime(const Options& options);
    bool initialize(std::string& error, ggml_backend_sched_eval_callback callback, void* data);
    bool initialize_context(std::string& error, ggml_backend_sched_eval_callback callback, void* data);

    Options options_;
    std::array<ggml_backend_dev_t, 2> devices_{};
    std::array<llama_model_tensor_buft_override, 2> overrides_{};
    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;
    ggml_threadpool_t threadpool_ = nullptr;
    // Borrowed model-page registrations survive same-model context changes.
    // Released explicitly after contexts drain and before model mappings die.
    std::shared_ptr<ExpertSourcePin> retained_source_pin_;
    std::shared_ptr<PackedExpertStore> retained_packed_;
    std::unique_ptr<ExpertPageWarm> page_warm_;
    const llama_vocab* vocab_ = nullptr;
    llama_pos position_ = 0;
    ggml_backend_t mtp_loading_backend_=nullptr;
    std::unique_ptr<MtpSidecar> mtp_weights_;
    std::unique_ptr<MtpDrafter> mtp_drafter_;
    bool mtp_cached_only_=false;
    std::array<float,16384> mtp_hidden_{};
};

}  // namespace strata::deepseek_v4
