#include "runtime.hpp"
#include "device_cache.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace strata::deepseek_v4 {
namespace {
struct BatchOwner {
    llama_batch value;
    explicit BatchOwner(int32_t count) : value(llama_batch_init(count, 0, 1)) {}
    ~BatchOwner() { llama_batch_free(value); }
};
}
int Runtime::decode_tokens(const llama_token* tokens, size_t count, llama_pos start_pos,
                          bool logits_last, std::string& error,bool all_logits,int purpose) {
    error.clear();
    if (!context_ || !tokens || count == 0 || count > llama_n_batch(context_) || start_pos < 0 ||
        uint64_t(start_pos) + count > llama_n_ctx(context_)) {
        error = "token batch exceeds context/batch bounds or has invalid input";
        return -1;
    }
    BatchOwner batch{int32_t(count)};
    if (!batch.value.token || !batch.value.pos || !batch.value.seq_id || !batch.value.logits) {
        error = "could not allocate token batch";
        return -1;
    }
    batch.value.n_tokens = int32_t(count);
    for (size_t index = 0; index < count; ++index) {
        if (tokens[index] < 0 || tokens[index] >= llama_vocab_n_tokens(vocab_)) {
            error = "token ID lies outside the model vocabulary";
            return -1;
        }
        batch.value.token[index] = tokens[index];
        batch.value.pos[index] = start_pos + llama_pos(index);
        batch.value.n_seq_id[index] = 1;
        batch.value.seq_id[index][0] = 0;
        batch.value.logits[index] = int8_t(all_logits || (logits_last && index + 1 == count));
    }
    // purpose3 is restricted to a fresh, explicitly provisioned four-position diagnostic context.
    if(purpose<0||purpose>3||(purpose==2&&(!glm_mtp_||count!=2))||
       (purpose==3&&(!glm_verify4_probe_||count!=4)))throw std::runtime_error("decode purpose invalid");
    // A four-row PP tail remains PP even while the diagnostic enables width4 TG.
    prepare_device_expert_cache(context_,purpose==1?(glm_verify4_probe_&&count==4?5:std::max<size_t>(2,count)):(purpose==2||purpose==3)?1:count);
    const int status = llama_decode(context_, batch.value);
    if (status != 0) {
        error = "canonical decode returned status " + std::to_string(status);
        return status;
    }
    if(glm_mtp_)glm_capture_hidden(count,start_pos);
    position_ = start_pos + llama_pos(count);
    return 0;
}

bool Runtime::evaluate(const std::vector<llama_token>& tokens, std::string& error, const std::atomic<bool>* cancel) {
    error.clear();
    if (tokens.empty()) {
        error = "empty input token sequence";
        return false;
    }
    if (uint64_t(position_) + tokens.size() > llama_n_ctx(context_)) {
        error = "input token sequence exceeds available context";
        return false;
    }
    for (size_t offset = 0; offset < tokens.size();) {
        if(cancel&&cancel->load()){error="request cancelled";return false;}
        const size_t count = std::min(size_t(llama_n_batch(context_)), tokens.size() - offset);
        const bool last = offset + count == tokens.size();
        if (decode_tokens(tokens.data() + offset, count, position_, last, error,false,(position_==0||tokens.size()>1)?1:0) != 0) return false;
        offset += count;
    }
    return true;
}


}
