#include "runtime.hpp"
#include "device_cache.hpp"

#include <cstring>
#include <limits>

namespace strata::deepseek_v4 {

bool Runtime::decode_image(const float* rows, size_t count, llama_pos start_pos,
                           bool logits_last, std::string& error) {
    error.clear();
    const int32_t width = llama_model_n_embd_inp(model_);
    if (!context_ || !rows || width <= 0 || count == 0 ||
        count > options_.n_ubatch || count > size_t(std::numeric_limits<int32_t>::max()) ||
        start_pos < 0 || uint64_t(start_pos) + count > llama_n_ctx(context_)) {
        error = "image embedding batch exceeds model context or ubatch";
        return false;
    }
    llama_batch batch = llama_batch_init(int32_t(count), width, 1);
    if (!batch.embd || !batch.pos || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        error = "could not allocate image embedding batch";
        return false;
    }
    batch.n_tokens = int32_t(count);
    std::memcpy(batch.embd, rows, count * size_t(width) * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        batch.pos[i] = start_pos + llama_pos(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = int8_t(logits_last && i + 1 == count);
    }
    // The DeepSeek4V projector emits a complete image token block. As in
    // mtmd_helper_decode_image_chunk, its tokens use bidirectional attention.
    struct CausalRestore {
        llama_context* ctx;
        ~CausalRestore() { llama_set_causal_attn(ctx, true); }
    } restore{context_};
    llama_set_causal_attn(context_, metadata("general.architecture") == "glm5-next");
    prepare_device_expert_cache(context_, count);
    const int status = llama_decode(context_, batch);
    llama_batch_free(batch);
    if (status != 0) {
        error = "image embedding decode returned status " + std::to_string(status);
        return false;
    }
    position_ = start_pos + llama_pos(count);
    return true;
}

}  // namespace strata::deepseek_v4
