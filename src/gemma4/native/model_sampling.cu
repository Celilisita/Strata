#include "strata/gemma4/native/model_sampling.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cmath>
namespace strata::gemma4::native {
namespace {
// Same cap*tanh(raw/cap) expression and finite flag as the admitted head pick.
// The sampler receives the real capped logits, without a CPU logits copy.
__global__ void cap_logits(float* logits, int count, float cap, unsigned* invalid) {
    const int id = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (id >= count) return;
    const float raw = logits[id];
    if (!isfinite(raw)) atomicOr(invalid, 1u);
    logits[id] = cap * tanhf(raw / cap);
}
// Preserve all earlier model/state failures. The shared sampler's export API
// assigns its own flag, so Gemma exports its pick with request-wide OR semantics.
__global__ void export_sampling(const kernels::SamplePick* pick, int* next,
                                unsigned* invalid) {
    const bool bad = pick->first_bad != -1;
    if (bad) atomicOr(invalid, 1u);
    *next = !*invalid ? pick->id : 0;
}
}
void finish_sampling_pick(const kernels::SamplePick* pick, int* next,
                          unsigned* invalid, cudaStream_t stream) {
    export_sampling<<<1, 1, 0, stream>>>(pick, next, invalid);
    ::strata::qwen35::native::check_cuda(cudaGetLastError(), "Gemma sampled pick and state flag");
}
void prepare_sampling_logits(float* logits, int vocabulary, float cap,
                             unsigned* invalid, cudaStream_t stream) {
    if (!logits || !invalid || !stream || vocabulary < 1 || vocabulary > 262144 ||
        !(cap > 0) || !std::isfinite(cap))
        throw std::invalid_argument("Gemma capped sampler geometry");
    cap_logits<<<(vocabulary + 255) / 256, 256, 0, stream>>>(logits, vocabulary, cap, invalid);
    ::strata::qwen35::native::check_cuda(cudaGetLastError(), "Gemma capped sampling logits");
}
} // namespace strata::gemma4::native
