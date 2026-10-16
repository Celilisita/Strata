#include "visual_state.hpp"
#include "session_state.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
namespace strata::qwen35moe {
void Runtime::initialize_visual() {
    const char* enabled = std::getenv("STRATA_QWEN35_VISION");
    if (!enabled || std::string(enabled) == "0") return;
    if (std::string(enabled) != "1" || spec_.rope_sections[0] != 11 || spec_.rope_sections[1] != 11 ||
        spec_.rope_sections[2] != 10 || spec_.rope_sections[3] != 0)
        throw std::invalid_argument("visual provider requires interleaved11/11/10/0 IMRoPE");
    visual_ = std::make_unique<VisualState>(max_context_);
}
void Runtime::set_visual_input(VisualInput input) {
    if (!visual_ || decode_phase_ || input.spans.empty() || input.spans.size() > 64 || input.positions.empty() ||
        input.positions.size() > size_t(max_context_)) throw std::invalid_argument("unprepared visual request");
    int previous_end = 0;
    for (const auto& span : input.spans) {
        const auto& image = span.image; const int64_t n = int64_t(image.width) * image.height;
        if (image.width < 1 || image.height < 1 || span.first < previous_end || n < 1 || n > int64_t(input.positions.size()) - span.first ||
            image.embedding_width != int(spec_.n_embd) || image.rows.size() != size_t(n) * spec_.n_embd ||
            std::any_of(image.rows.begin(), image.rows.end(), [](float x) { return !std::isfinite(x); }))
            throw std::invalid_argument("invalid visual span");
        previous_end = span.first + int(n);
    }
    for (const auto& position : input.positions) for (int value : position)
        if (value < 0 || value > 1000000000) throw std::invalid_argument("invalid visual rotary position");
    if (!visual_->input.spans.empty())
        cuda_check(cudaStreamSynchronize(stream_), "replace visual input after prior consumers");
    auto& upload = visual_->upload;
    std::copy(input.positions.begin(), input.positions.end(), upload.begin());
    int logical = *(std::max_element)(input.positions.back().begin(), input.positions.back().end()) + 1;
    for (size_t i = input.positions.size(); i < upload.size(); ++i, ++logical) upload[i] = {logical, logical, logical};
    visual_->input = std::move(input);
    cuda_check(cudaMemcpyAsync(visual_->table.data(), upload.data(), visual_->table.bytes(),
        cudaMemcpyHostToDevice, stream_), "visual position table");
    visual_h2d_pending_ += visual_->table.bytes();
}
void Runtime::collect_visual_metrics(Metrics& metrics) {
    metrics.visual_h2d += visual_h2d_pending_; visual_h2d_pending_ = 0;
}
void Runtime::clear_visual_input() { if (visual_) visual_->input = VisualInput{}; }
const int32_t* Runtime::visual_table(int session) const {
    const VisualState* state = visual_.get();
    if (session >= 0 && sessions_ && sessions_->active != session) state = sessions_->slots.at(size_t(session)).visual.get();
    return state && !state->input.spans.empty() ? state->table.as<int32_t>() : nullptr;
}
void Runtime::load_embedding(int token, int position, float* output, const TensorInfo* source,
    DeviceBuffer& scratch, size_t row_bytes, int session) {
    const VisualState* state = visual_.get();
    if (session >= 0 && sessions_ && sessions_->active != session) state = sessions_->slots.at(size_t(session)).visual.get();
    if (state) for (const auto& span : state->input.spans) {
        const int rows = span.image.width * span.image.height;
        if (position >= span.first && position - span.first < rows) {
            const float* row = span.image.rows.data() + size_t(position - span.first) * spec_.n_embd;
            cuda_check(cudaMemcpyAsync(output, row, size_t(spec_.n_embd) * sizeof(float),
                cudaMemcpyHostToDevice, stream_), "visual embedding row");
            visual_h2d_pending_ += size_t(spec_.n_embd) * sizeof(float); return;
        }
    }
    const auto* row = gguf_.tensor_data(*source) + size_t(token) * row_bytes;
    cuda_check(cudaMemcpyAsync(scratch.data(), row, row_bytes, cudaMemcpyHostToDevice, stream_), "embedding row");
    kernels::dequant_f32(int(source->type), scratch.data(), 0, 1, int(spec_.n_embd), output, stream_);
}
} // namespace strata::qwen35moe
