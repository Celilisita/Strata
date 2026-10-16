#include "strata/qwen35moe/runtime.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35moe {
void Runtime::load_prefill_embeddings(const int* tokens, int count, float* output) {
    const char* raw = std::getenv("STRATA_QWEN35_PP_BATCH_EMBEDDING");
    if (raw && std::strcmp(raw, "0") && std::strcmp(raw, "1"))
        throw std::invalid_argument("PP_BATCH_EMBEDDING requires0 or1");
    for (int t = 0; t < count; ++t)
        if (tokens[t] < 0 || tokens[t] >= int(spec_.n_vocab))
            throw std::out_of_range("Qwen35 prefill token");
    // The quantization workspace is dead until the first layer. Stream ordering
    // protects each upload from the preceding dequantization and later reuse.
    const size_t capacity = batch_.input_q8.bytes() / embedding_row_bytes_;
    const size_t bytes = size_t(count) * embedding_row_bytes_;
    const bool enabled = raw && !std::strcmp(raw, "1");
    if (!enabled || visual_table() || !capacity || bytes > (size_t(64) << 20)) {
        for (int t = 0; t < count; ++t)
            load_embedding(tokens[t], t, output + size_t(t) * spec_.n_embd,
                spec_.embedding, work_.emb_row, embedding_row_bytes_);
        return;
    }
    auto& rows = batch_.embedding_rows;
    rows.resize(bytes);
    const auto* source = gguf_.tensor_data(*spec_.embedding);
    for (int t = 0; t < count; ++t)
        std::memcpy(rows.data() + size_t(t) * embedding_row_bytes_,
            source + size_t(tokens[t]) * embedding_row_bytes_, embedding_row_bytes_);
    for (int first = 0; first < count;) {
        const int n = int((std::min)(capacity, size_t(count - first)));
        cuda_check(cudaMemcpyAsync(batch_.input_q8.data(),
            rows.data() + size_t(first) * embedding_row_bytes_, size_t(n) * embedding_row_bytes_,
            cudaMemcpyHostToDevice, stream_), "prefill embedding rows");
        kernels::dequant_f32(int(spec_.embedding->type), batch_.input_q8.data(),
            0, n, int(spec_.n_embd), output + size_t(first) * spec_.n_embd, stream_);
        first += n;
    }
}
} // namespace strata::qwen35moe
