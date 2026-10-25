#include <strata/kernels/radix_sampler.hpp>
#include "top20_selection_policy.hpp"
#include "top20_prepare_fusion_policy.hpp"
#include <climits>
#include "strata/kernels/sampler_math.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <cstring>
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
constexpr int TopVocab=248320,TopBlock=4096,TopKeep=20,TopBlocks=(TopVocab+TopBlock-1)/TopBlock;
// Same float radix transform as CUB, with signed zeros canonicalized equal.
// Descending low bits impose the original stable lower-token-ID tie order.
__device__ unsigned long long rank_key(float x,unsigned id){
    unsigned bits=__float_as_uint(x);if(!(bits&0x7fffffffU))bits=0;
    const unsigned order=bits^((bits&0x80000000U)?0xffffffffU:0x80000000U);
    return (static_cast<unsigned long long>(order)<<32)|static_cast<unsigned>(~id);
}
__global__ void block_top20(const float*scores,unsigned long long*temporary){
    using Sort=cub::BlockRadixSort<unsigned long long,256,16>;
    __shared__ typename Sort::TempStorage storage;unsigned long long keys[16];
    #pragma unroll
    for(int i=0;i<16;++i){const unsigned id=blockIdx.x*TopBlock+threadIdx.x*16+i;keys[i]=id<TopVocab?rank_key(scores[id],id):0ULL;}
    Sort(storage).SortDescending(keys);
    #pragma unroll
    for(int i=0;i<16;++i){const int rank=threadIdx.x*16+i;if(rank<TopKeep)temporary[blockIdx.x*TopKeep+rank]=keys[i];}
}
// Fast only for exact currentTop20 with h0 and no penalties. Every live logit still contributes first_bad.
__global__ void block_top20_fused(const float*logits,unsigned long long*temporary,Control*control){
    using Sort=cub::BlockRadixSort<unsigned long long,256,16>;
    __shared__ typename Sort::TempStorage storage;unsigned long long keys[16];
    #pragma unroll
    for(int i=0;i<16;++i){const unsigned id=blockIdx.x*TopBlock+threadIdx.x*16+i;
        const float x=id<TopVocab?logits[id]:0.f;
        if(id<TopVocab&&!isfinite(x))atomicMin(&control->first_bad,int(id));
        keys[i]=id<TopVocab?rank_key(x,id):0ULL;}
    Sort(storage).SortDescending(keys);
    #pragma unroll
    for(int i=0;i<16;++i){const int rank=threadIdx.x*16+i;if(rank<TopKeep)temporary[blockIdx.x*TopKeep+rank]=keys[i];}
}
__global__ void merge_top20(const float*scores,const unsigned long long*temporary,float*sorted_scores,int*sorted_ids){
    using Sort=cub::BlockRadixSort<unsigned long long,256,8>;
    __shared__ typename Sort::TempStorage storage;unsigned long long keys[8];
    #pragma unroll
    for(int i=0;i<8;++i){const int at=threadIdx.x*8+i;keys[i]=at<TopBlocks*TopKeep?temporary[at]:0ULL;}
    Sort(storage).SortDescending(keys);
    // Input ranks borrow sorted_ids. Finish every read before overwriting its prefix.
    __syncthreads();
    #pragma unroll
    for(int i=0;i<8;++i){const int rank=threadIdx.x*8+i;if(rank<TopKeep){const int id=int(~static_cast<unsigned>(keys[i]));sorted_scores[rank]=scores[id];sorted_ids[rank]=id;}}
}
static_assert(TopBlocks==61&&TopBlocks*TopKeep==1220&&TopBlocks*TopKeep*8<TopVocab*4,"bounded existing sorted-ID scratch");
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
    int n, int prefix_length, SamplerParams p, const Control* c, SamplePick* result, const int* failed,
    const uint64_t* counter, unsigned offset) {
    if (counter) p.counter = *counter + offset;
    const int failure = failed ? *failed : 0;
    const int bad = failure ? -2 - failure : (c->first_bad == n ? -1 : c->first_bad);
    int chosen = 0;
    const double total = prefix[prefix_length - 1];
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
__global__ void export_pick(const SamplePick* pick, int* id, unsigned* invalid) {
    *id = pick->first_bad == -1 ? pick->id : 0;
    *invalid = unsigned(pick->first_bad != -1);
}
} // namespace
namespace {
std::atomic<uint64_t> sampler_calls{0}, bounded_calls{0};
}
bool radix_short_prefix_requested() {
    static const bool requested = [] {
        const char* text = std::getenv("STRATA_DENSE_SHORT_PREFIX");
        if (!text || std::strcmp(text, "0") == 0) return false;
        if (std::strcmp(text, "1") == 0) return true;
        throw std::invalid_argument("STRATA_DENSE_SHORT_PREFIX must be 0 or 1");
    }();
    return requested;
}
bool radix_short_prefix_enabled() {
    static const bool admitted = [] {
        if (!radix_short_prefix_requested()) return false;
        int device = 0; check(cudaGetDevice(&device));
        cudaDeviceProp caps{}; check(cudaGetDeviceProperties(&caps, device));
        return caps.major == 12 && caps.minor == 0;
    }();
    return admitted;
}
uint64_t radix_short_prefix_host_calls() { return sampler_calls.load(std::memory_order_relaxed); }
uint64_t radix_short_prefix_bounded_host_calls() { return bounded_calls.load(std::memory_order_relaxed); }

namespace {
struct Top20Admission {
    bool requested=false,capable=false;
    int binary_version=0,blocks_per_sm=0,registers=0;
    size_t local_bytes=0;
};
std::atomic<const Top20Admission*> top20_published{nullptr};
std::atomic<uint64_t> top20_calls{0},top20_fallbacks{0};
std::atomic<int> top20_last_vocab{0},top20_last_k{0};
const Top20Admission& top20_admission() {
    static const Top20Admission result=[] {
        Top20Admission r;
        const char* text=std::getenv("STRATA_DENSE_TOP20_SELECTION");
        if(!text||std::strcmp(text,"0")==0)return r;
        if(std::strcmp(text,"1")!=0)
            throw std::invalid_argument("STRATA_DENSE_TOP20_SELECTION must be 0 or 1");
        r.requested=true;
        int device=0;check(cudaGetDevice(&device));cudaDeviceProp caps{};
        check(cudaGetDeviceProperties(&caps,device));
        if(caps.major!=12||caps.minor!=0)return r;
        r.blocks_per_sm=INT_MAX;
        auto note=[&](auto kernel,int threads) {
            cudaFuncAttributes a{};int blocks=0;
            check(cudaFuncGetAttributes(&a,kernel));
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,0));
            r.binary_version=a.binaryVersion;
            r.blocks_per_sm=(std::min)(r.blocks_per_sm,blocks);
            r.registers=(std::max)(r.registers,a.numRegs);
            r.local_bytes=(std::max)(r.local_bytes,a.localSizeBytes);
            if(a.binaryVersion!=120)r.blocks_per_sm=0;
        };
        note(init,1);note(counts,256);note(prepare,256);note(raw_mass,256);
        note(cut_support,1);note(final_mass,256);note(draw,1);note(export_pick,1);
        note(block_top20,256);note(merge_top20,256);
        r.capable=r.blocks_per_sm>=1&&r.local_bytes==0;
        return r;
    }();
    top20_published.store(&result,std::memory_order_release);
    return result;
}
} // namespace
Top20SelectionStats top20_selection_stats() {
    Top20SelectionStats s;
    if(const auto* p=top20_published.load(std::memory_order_acquire)) {
        s.requested=p->requested;s.capable=p->capable;
        s.binary_version=p->binary_version;s.blocks_per_sm=p->blocks_per_sm;
        s.registers=p->registers;s.local_bytes=p->local_bytes;
    }
    s.host_calls=top20_calls.load(std::memory_order_relaxed);
    s.fallback_calls=top20_fallbacks.load(std::memory_order_relaxed);
    s.admitted=s.host_calls!=0;
    s.last_vocabulary=top20_last_vocab.load(std::memory_order_relaxed);
    s.last_top_k=top20_last_k.load(std::memory_order_relaxed);
    return s;
}

namespace {
std::atomic<const Top20PrepareFusionStats*> prepare_fusion_published{nullptr};
std::atomic<uint64_t> prepare_fusion_calls{0},prepare_fusion_fallbacks{0};
std::atomic<int> prepare_fusion_last_vocab{0},prepare_fusion_last_k{0};
bool prepare_fusion_requested(){
    static const bool requested=[](){const char*raw=std::getenv("STRATA_DENSE_TOP20_PREPARE_FUSION");
        const std::string flag=raw?raw:"0";
        if(flag!="0"&&flag!="1")throw std::invalid_argument("Top20 prepare fusion must be0or1");return flag=="1";}();
    return requested;
}
const Top20PrepareFusionStats &prepare_fusion_admission(){
    static const Top20PrepareFusionStats result=[](){
        Top20PrepareFusionStats r;r.requested=prepare_fusion_requested();if(!r.requested)return r;
        int device=0;check(cudaGetDevice(&device));cudaDeviceProp caps{};check(cudaGetDeviceProperties(&caps,device));
        if(caps.major!=12||caps.minor!=0)return r;
        cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,block_top20_fused));
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.blocks_per_sm,block_top20_fused,256,0));
        r.binary_version=attributes.binaryVersion;r.registers=attributes.numRegs;r.local_bytes=attributes.localSizeBytes;
        if(r.binary_version!=120||r.blocks_per_sm<1||r.local_bytes)
            throw std::runtime_error("Top20 prepare fusion requires qualifiedSM120 zero-local kernel");
        r.capable=true;return r;
    }();prepare_fusion_published.store(&result,std::memory_order_release);return result;
}
}
Top20PrepareFusionStats top20_prepare_fusion_stats(){
    Top20PrepareFusionStats r;
    if(const auto*p=prepare_fusion_published.load(std::memory_order_acquire))r=*p;
    r.host_calls=prepare_fusion_calls.load(std::memory_order_relaxed);
    r.fallback_calls=prepare_fusion_fallbacks.load(std::memory_order_relaxed);r.admitted=r.host_calls!=0;
    r.last_vocabulary=prepare_fusion_last_vocab.load(std::memory_order_relaxed);
    r.last_top_k=prepare_fusion_last_k.load(std::memory_order_relaxed);return r;
}

struct RadixSampler::State {
    int n;
    bool short_prefix,top20,prepare_fusion,prepare_fusion_request;
    Allocation scores, sorted_scores, ids, sorted_ids, occurrences;
    Allocation mass, prefix, control, pick;
    std::unique_ptr<Allocation> temporary;
    size_t sort_bytes = 0, scan_bytes = 0;
    explicit State(int vocab) : n(vocab), short_prefix(radix_short_prefix_enabled()),
        top20(top20_admission().capable&&vocab==TopVocab&&short_prefix),
        prepare_fusion(prepare_fusion_admission().capable&&top20),prepare_fusion_request(prepare_fusion_requested()),scores(size_t(n) * sizeof(float)),
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
    sample_counter(logits, history, length, p, raw_stream, nullptr, 0, failed);
}
void RadixSampler::sample_counter(const float* logits, const int* history, int length,
    const SamplerParams& p, void* raw_stream, const uint64_t* counter,
    unsigned offset, const int* failed) {
    validate(p);
    if (!logits || length < 0 || (length && !history))
        throw std::invalid_argument("invalid GPU sampler inputs");
    auto& s = *state_;
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const int blocks = (s.n + 255) / 256;
    const int h = (std::min)(length, p.penalty_last_n);
    const int k = p.top_k ? (std::min)(s.n, p.top_k) : s.n;
    init<<<1, 1, 0, stream>>>(s.control.as<Control>(), s.n);
    const bool fast=s.prepare_fusion&&s.top20&&p.top_k==TopKeep&&h==0&&
        p.penalty_repeat==1.f&&p.penalty_freq==0.f&&p.penalty_present==0.f;
    if(!fast){
        // Identical fallback setup: never consume old occurrences/scores/identity IDs after fast requests.
        check(cudaMemsetAsync(s.occurrences.p, 0, s.occurrences.bytes, stream));
        if (h) counts<<<(h + 255) / 256, 256, 0, stream>>>(history + length - h, h, s.occurrences.as<int>(), s.n);
        prepare<<<blocks, 256, 0, stream>>>(logits, s.occurrences.as<int>(), s.scores.as<float>(),
            s.ids.as<int>(), s.n, p, s.control.as<Control>());
    }
    if(s.prepare_fusion_request){
        if(fast)prepare_fusion_calls.fetch_add(1,std::memory_order_relaxed);
        else prepare_fusion_fallbacks.fetch_add(1,std::memory_order_relaxed);
        prepare_fusion_last_vocab.store(s.n,std::memory_order_relaxed);prepare_fusion_last_k.store(p.top_k,std::memory_order_relaxed);
    }
    if(s.top20&&p.top_k==TopKeep) {
        auto* rank_scratch=s.sorted_ids.as<unsigned long long>();
        const float*selected_source=fast?logits:s.scores.as<float>();
        if(fast)block_top20_fused<<<TopBlocks,256,0,stream>>>(logits,rank_scratch,s.control.as<Control>());
        else block_top20<<<TopBlocks,256,0,stream>>>(selected_source,rank_scratch);
        check(cudaGetLastError());
        merge_top20<<<1,256,0,stream>>>(selected_source,rank_scratch,
            s.sorted_scores.as<float>(),s.sorted_ids.as<int>());
        check(cudaGetLastError());
        top20_calls.fetch_add(1,std::memory_order_relaxed);
    }else{
        check(cub::DeviceRadixSort::SortPairsDescending(s.temporary->p, s.sort_bytes, s.scores.as<float>(),
            s.sorted_scores.as<float>(), s.ids.as<int>(), s.sorted_ids.as<int>(), s.n, 0, 32, stream));
        top20_fallbacks.fetch_add(1,std::memory_order_relaxed);
    }
    top20_last_vocab.store(s.n,std::memory_order_relaxed);
    top20_last_k.store(p.top_k,std::memory_order_relaxed);
    const int prefix_length=s.short_prefix?k:s.n, prefix_blocks=(prefix_length+255)/256;
    sampler_calls.fetch_add(1, std::memory_order_relaxed);
    if (prefix_length < s.n) bounded_calls.fetch_add(1, std::memory_order_relaxed);
    raw_mass<<<prefix_blocks,256,0,stream>>>(s.sorted_scores.as<float>(),s.mass.as<double>(),prefix_length,k);
    size_t available_scan_bytes=s.scan_bytes;
    check(cub::DeviceScan::InclusiveSum(s.temporary->p,available_scan_bytes,s.mass.as<double>(),s.prefix.as<double>(),prefix_length,stream));
    cut_support<<<1, 1, 0, stream>>>(s.sorted_scores.as<float>(), s.prefix.as<double>(), k, p, s.control.as<Control>());
    final_mass<<<prefix_blocks,256,0,stream>>>(s.sorted_scores.as<float>(),s.mass.as<double>(),prefix_length,p,s.control.as<Control>());
    available_scan_bytes=s.scan_bytes;
    check(cub::DeviceScan::InclusiveSum(s.temporary->p,available_scan_bytes,s.mass.as<double>(),s.prefix.as<double>(),prefix_length,stream));
    draw<<<1, 1, 0, stream>>>(logits, s.sorted_ids.as<int>(), s.prefix.as<double>(), s.n, prefix_length, p,
        s.control.as<Control>(), s.pick.as<SamplePick>(), failed, counter, offset);
    check(cudaGetLastError());
}
void RadixSampler::write_pick(int* id, unsigned* invalid, void* raw_stream) const {
    if (!id || !invalid) throw std::invalid_argument("invalid sampler pick destination");
    export_pick<<<1, 1, 0, static_cast<cudaStream_t>(raw_stream)>>>(state_->pick.as<SamplePick>(), id, invalid);
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
