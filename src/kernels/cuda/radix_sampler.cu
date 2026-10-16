#include "strata/kernels/radix_sampler.hpp"
#include "strata/kernels/sampler_math.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("GPU sampler: ") + cudaGetErrorString(e));
}
struct Allocation {
    void* p = nullptr;
    size_t bytes = 0;
    explicit Allocation(size_t n = 0) : bytes(n) { if (n) check(cudaMalloc(&p, n)); }
    ~Allocation() { if (p) cudaFree(p); }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
};
struct Control { int first_bad, kept; };
__global__ void init(Control* c, int n) {
    c->first_bad = n; c->kept = 0;
}
__global__ void counts(const int* history, int length, int* output, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < length && history[i] >= 0 && history[i] < n) atomicAdd(output + history[i], 1);
}
__global__ void prepare(const float* logits, const int* occurrence, float* scores,
    int* ids, int n, SamplerParams p, Control* c) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float x = sampler_math::penalize(logits[i], occurrence[i], p);
    if (!isfinite(logits[i]) || !isfinite(x)) atomicMin(&c->first_bad, i);
    scores[i] = x; ids[i] = i;
}
__global__ void raw_mass(const float* scores, double* mass, int n, int k) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) mass[i] = i < k ? exp(double(scores[i]) - double(scores[0])) : 0.0;
}
__global__ void cut_support(const float* scores, const double* prefix, int k,
    SamplerParams p, Control* c) {
    int keep = k;
    if (p.greedy || p.temperature <= 0) keep = 1;
    else {
        if (p.top_p < 1) {
            const double threshold = double(p.top_p) * prefix[k - 1];
            int lo = 0, hi = k;
            while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (prefix[mid] < threshold) lo = mid + 1; else hi = mid;
            }
            keep = min(k, max(min(k, p.min_keep), lo + 1));
        }
        if (p.min_p > 0) {
            const double threshold = double(scores[0]) + log(double(p.min_p));
            int lo = 0, hi = keep;
            while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (double(scores[mid]) >= threshold) lo = mid + 1; else hi = mid;
            }
            keep = min(keep, max(min(keep, p.min_keep), lo));
        }
    }
    c->kept = keep;
}
__global__ void final_mass(const float* scores, double* mass, int n,
    SamplerParams p, const Control* c) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) mass[i] = i < c->kept ? exp((double(scores[i]) - double(scores[0])) /
        ((p.greedy || p.temperature <= 0) ? 1.0 : double(p.temperature))) : 0.0;
}
__global__ void draw(const float* logits, const int* ids, const double* prefix,
    int n, SamplerParams p, const Control* c, SamplePick* result, const int* failed) {
    const int failure = failed ? *failed : 0;
    const int bad = failure ? -2 - failure : (c->first_bad == n ? -1 : c->first_bad);
    int chosen = 0;
    const double total = prefix[n - 1];
    if (!p.greedy && p.temperature > 0 && isfinite(total) && total > 0) {
        const double target = double(sampler_math::uniform(p.seed, p.counter)) * total;
        int lo = 0, hi = c->kept;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (prefix[mid] <= target) lo = mid + 1; else hi = mid;
        }
        chosen = min(lo, c->kept - 1);
    }
    const double mass = prefix[chosen] - (chosen ? prefix[chosen - 1] : 0.0);
    *result = {ids[chosen], bad, logits[ids[chosen]], float(mass / total)};
}
} // namespace

struct RadixSampler::State {
    int n;
    Allocation scores, sorted_scores, ids, sorted_ids, occurrences;
    Allocation mass, prefix, control, pick;
    std::unique_ptr<Allocation> temporary;
    size_t sort_bytes = 0, scan_bytes = 0;
    explicit State(int vocab) : n(vocab), scores(size_t(n) * sizeof(float)),
        sorted_scores(scores.bytes), ids(size_t(n) * sizeof(int)), sorted_ids(ids.bytes),
        occurrences(ids.bytes), mass(size_t(n) * sizeof(double)), prefix(mass.bytes),
        control(sizeof(Control)), pick(sizeof(SamplePick)) {
        check(cub::DeviceRadixSort::SortPairsDescending(nullptr, sort_bytes, scores.as<float>(),
            sorted_scores.as<float>(), ids.as<int>(), sorted_ids.as<int>(), n));
        check(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, mass.as<double>(), prefix.as<double>(), n));
        temporary = std::make_unique<Allocation>((std::max)(sort_bytes, scan_bytes));
    }
};
RadixSampler::RadixSampler(int vocabulary) {
    if (vocabulary < 1) throw std::invalid_argument("sampler vocabulary must be positive");
    state_ = std::make_unique<State>(vocabulary);
}
RadixSampler::~RadixSampler() = default;
void RadixSampler::validate(const SamplerParams& p) {
    if (p.top_k < 0 || p.min_keep < 1 || p.penalty_last_n < 0 ||
        !std::isfinite(p.top_p) || p.top_p < 0 || p.top_p > 1 ||
        !std::isfinite(p.min_p) || p.min_p < 0 || p.min_p > 1 ||
        !std::isfinite(p.temperature) || !std::isfinite(p.penalty_repeat) || p.penalty_repeat <= 0 ||
        !std::isfinite(p.penalty_freq) || !std::isfinite(p.penalty_present))
        throw std::invalid_argument("invalid GPU sampler parameters");
}
void RadixSampler::sample(const float* logits, const int* history, int length,
    const SamplerParams& p, void* raw_stream, const int* failed) {
    validate(p);
    if (!logits || length < 0 || (length && !history))
        throw std::invalid_argument("invalid GPU sampler inputs");
    auto& s = *state_;
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const int blocks = (s.n + 255) / 256;
    const int h = (std::min)(length, p.penalty_last_n);
    const int k = p.top_k ? (std::min)(s.n, p.top_k) : s.n;
    init<<<1, 1, 0, stream>>>(s.control.as<Control>(), s.n);
    check(cudaMemsetAsync(s.occurrences.p, 0, s.occurrences.bytes, stream));
    if (h) counts<<<(h + 255) / 256, 256, 0, stream>>>(history + length - h, h, s.occurrences.as<int>(), s.n);
    prepare<<<blocks, 256, 0, stream>>>(logits, s.occurrences.as<int>(), s.scores.as<float>(),
        s.ids.as<int>(), s.n, p, s.control.as<Control>());
    check(cub::DeviceRadixSort::SortPairsDescending(s.temporary->p, s.sort_bytes, s.scores.as<float>(),
        s.sorted_scores.as<float>(), s.ids.as<int>(), s.sorted_ids.as<int>(), s.n, 0, 32, stream));
    raw_mass<<<blocks, 256, 0, stream>>>(s.sorted_scores.as<float>(), s.mass.as<double>(), s.n, k);
    check(cub::DeviceScan::InclusiveSum(s.temporary->p, s.scan_bytes, s.mass.as<double>(), s.prefix.as<double>(), s.n, stream));
    cut_support<<<1, 1, 0, stream>>>(s.sorted_scores.as<float>(), s.prefix.as<double>(), k, p, s.control.as<Control>());
    final_mass<<<blocks, 256, 0, stream>>>(s.sorted_scores.as<float>(), s.mass.as<double>(), s.n, p, s.control.as<Control>());
    check(cub::DeviceScan::InclusiveSum(s.temporary->p, s.scan_bytes, s.mass.as<double>(), s.prefix.as<double>(), s.n, stream));
    draw<<<1, 1, 0, stream>>>(logits, s.sorted_ids.as<int>(), s.prefix.as<double>(), s.n, p,
        s.control.as<Control>(), s.pick.as<SamplePick>(), failed);
    check(cudaGetLastError());
}
const SamplePick* RadixSampler::result() const { return state_->pick.as<SamplePick>(); }
const int* RadixSampler::sorted_ids() const { return state_->sorted_ids.as<int>(); }
const double* RadixSampler::cumulative() const { return state_->prefix.as<double>(); }
const int* RadixSampler::kept_count() const { return &state_->control.as<Control>()->kept; }
size_t RadixSampler::workspace_bytes() const {
    const auto& s = *state_;
    return s.scores.bytes + s.sorted_scores.bytes + s.ids.bytes + s.sorted_ids.bytes +
        s.occurrences.bytes + s.mass.bytes + s.prefix.bytes + s.control.bytes + s.pick.bytes + s.temporary->bytes;
}
} // namespace strata::kernels
