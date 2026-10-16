#include "native_tg_cuda.cuh"
#include "native_tg_timing.cuh"
#include "native_tg_trace.hpp"
#include "router_prediction_cuda.cuh"
#include "router_prefetch.hpp"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include <cuda/atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
namespace strata::deepseek_v4 {
namespace {
using Word = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
__host__ __device__ uint32_t& plain(volatile uint32_t& v) { return const_cast<uint32_t&>(v); }
__device__ uint64_t clock_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
// GPU-only diagnostic. Both physical positions of one verification share
// timestamps; sum one position per proposal. Join end observes block zero.
__device__ void phase_stamp(NativePhaseClock* trace,const uint32_t* epoch,int layer,int tokens,int stage) {
    if(!trace||layer>=43)return;
    const uint32_t first=*epoch-tokens+1;const auto now=clock_ns();
    for(int t=0;t<tokens;++t)if(first+t&&first+t<=8192){auto& r=trace[(first+t-1)*43+layer];
        if(stage==0)r.publish=now;else if(stage==1)r.join_begin=now;else if(stage==2)r.cpu_ready=now;else r.join_end=now;}
}
__device__ void phase_finish(NativePhaseClock* trace,const uint32_t* epoch,int layer,int tokens) {
    if(trace&&layer<43&&blockIdx.x==0){__syncthreads();if(threadIdx.x==0)phase_stamp(trace,epoch,layer,tokens,3);}
}
// state: miss mask, ticket, original-rank -> GPU-rank inverse (six words).
__global__ void publish(const int32_t* directory, const int32_t* ids, const float* x, int32_t* slots,
    uint32_t* state, int32_t* status, unsigned long long* counters, uint32_t* frequency,
    DeviceMissPacket* packet, int layer, float clamp, bool cpu_only, bool draft_skip, int tokens, size_t score_count, uint32_t* trace_epoch, NativeRouteRecord* trace,NativePhaseClock* phase_trace,
    const float* route_weights,unsigned long long* weight_buckets) {
    __shared__ uint32_t missing, ticket;
    __shared__ int initial[12], order[12];
    const int routes=tokens*6;
    if (threadIdx.x == 0) {
        atomicAdd(counters, 1ull);
        missing = 0; ticket = 0;
        int hits = 0;
        for (int rank = 0; rank < routes; ++rank) {
            const int id = ids[rank];
            initial[rank] = id >= 0 && id < 256 ? directory[id] : -1;
            if (id < 0 || id >= 256) atomicExch(status, 1);
            if (initial[rank] < 0) missing |= 1u << rank;
            else ++hits;
        }
        if(weight_buckets&&layer<43){
            float sums[2]{0,0};
            for(int rank=0;rank<routes;++rank)sums[rank/6]+=route_weights[rank];
            unsigned long long counts[4]{0,0,0,0};
            for(int rank=0;rank<routes;++rank){
                if(initial[rank]>=0)continue;
                bool seen=false;float maximum=0;
                for(int other=0;other<routes;++other)if(ids[other]==ids[rank]){
                    if(other<rank){seen=true;break;}
                    const float sum=sums[other/6];
                    if(sum>0)maximum=fmaxf(maximum,route_weights[other]/sum);
                }
                if(seen)continue;
                ++counts[0];if(maximum<.03f)++counts[1];
                if(maximum<.05f)++counts[2];if(maximum<.08f)++counts[3];
            }
            for(int i=0;i<4;++i)if(counts[i])atomicAdd(weight_buckets+i,counts[i]);
        }
        atomicAdd(counters + 2, static_cast<unsigned long long>(hits));
        if (!missing && !*status) atomicAdd(counters + 1, 1ull);
        if (frequency && !*status) {
            for (int rank = 0; rank < routes; ++rank) {
                if (++frequency[score_count] == 65536) {
                    for (size_t key = 0; key < score_count; ++key) frequency[key] /= 2;
                    frequency[score_count] = 0;
                }
                auto& score = frequency[layer * 256 + ids[rank]];
                if (score < 65535) ++score;
            }
        }
        if (!missing || tokens==2) {
            for (int rank = 0; rank < routes; ++rank) { slots[rank] = initial[rank];state[2+rank]=rank;order[rank]=rank; }
        } else {
            int next = 0;
            for (int rank = 0; rank < routes; ++rank) if (initial[rank] >= 0) order[next++] = rank;
            for (int rank = 0; rank < routes; ++rank) if (initial[rank] < 0) order[next++] = rank;
            for (int gpu_rank = 0; gpu_rank < routes; ++gpu_rank) {
                const int rank = order[gpu_rank];
                state[2 + rank] = gpu_rank;
                slots[gpu_rank] = initial[rank] >= 0 ? initial[rank] : (cpu_only ? -1 : -2);
            }
        }
        if(trace_epoch && layer<43) {
            if(layer==0)*trace_epoch+=tokens;
            const uint32_t first=*trace_epoch-tokens+1;
            for(int token=0;token<tokens;++token){const uint32_t epoch=first+token;
                if(trace&&epoch&&epoch<=8192){auto& record=trace[(epoch-1)*43+layer];
                    for(int rank=0;rank<6;++rank)record.ids[rank]=ids[token*6+rank];
                    record.missing=(missing>>(token*6))&63u;record.epoch=epoch;}}
        }
        state[0] = missing; state[1] = 0;
        phase_stamp(phase_trace,trace_epoch,layer,tokens,0);
        if (missing && !*status && !draft_skip) {
            ticket = Word(plain(packet->sequence)).load(cuda::memory_order_relaxed) + 1;
            if (!ticket) { atomicExch(status, 1); ticket = 0; }
        }
    }
    __syncthreads();
    if (!ticket) {
        if (*status && threadIdx.x < routes) slots[threadIdx.x] = -3;
        return; // All hit never reads or writes any mapped packet field.
    }
    for (int i = threadIdx.x; i < 4096*tokens; i += blockDim.x) packet->activation[i] = x[i];
    if (frequency && !cpu_only)
        for (size_t key = threadIdx.x; key <= score_count; key += blockDim.x)
            packet->frequency[key] = frequency[key];
    if (threadIdx.x < routes) {
        packet->ids[threadIdx.x] = ids[threadIdx.x];
        packet->gpu_order[threadIdx.x] = order[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        packet->native_request = tokens==2 ? 3 : (cpu_only ? 2 : 1); packet->layer = layer; packet->clamp = clamp;
        if (cpu_only) { packet->cpu_mask = missing; packet->cpu_error = 0; packet->error = 0; }
        packet->gpu_slots = reinterpret_cast<uintptr_t>(slots);
        state[1] = ticket;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        Word(plain(packet->sequence)).store(ticket, cuda::memory_order_release);
        atomicAdd(counters + 3, 1ull);
        if (cpu_only) return; // GPU hits and shared math run while CPU reads missing weights.
        // Wait without holding any expert bank in this kernel's arguments.
        // WDDM may serialize a bank upload against a running MMVQ that names
        // that same allocation, even when its pending rank has not read it.
        // Publish the ready slots from mapped metadata only after real DMA.
        const uint64_t begin = clock_ns();
        while (Word(plain(packet->response)).load(cuda::memory_order_acquire) != ticket) {
            if (clock_ns() - begin >= 1500000000ull) {
                atomicExch(status, 1); atomicAdd(counters + 4, 1ull); break;
            }
            __nanosleep(1000);
        }
        if (packet->error) atomicExch(status, 1);
        for (int rank = 0; rank < routes; ++rank)
            slots[rank] = *status ? -3 : packet->reply_slots[rank];
    }
}
__global__ void join(const float* down, const float* weights, float* output, const uint32_t* state,
    int32_t* status, unsigned long long* counters, DeviceMissPacket* packet, const float* shared, bool cpu_only, int tokens,
    NativePhaseClock* phase_trace,const uint32_t* trace_epoch,int layer,bool draft_skip) {
    if(blockIdx.x==0&&threadIdx.x==0)phase_stamp(phase_trace,trace_epoch,layer,tokens,1);
    // All-hit keeps original rank order. Bypass CPU masks, popcounts, inverse
    // gathers, mapped ticket logic and the block barrier entirely.
    if (!state[0]) {
        if(blockIdx.x==0&&threadIdx.x==0)phase_stamp(phase_trace,trace_epoch,layer,tokens,2);
        for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < 4096*tokens; row += gridDim.x * blockDim.x) {
            float value = 0;
#pragma unroll
            for (int rank = 0; rank < 6; ++rank) value += weights[(row/4096)*6+rank] * down[((row/4096)*6+rank)*4096 + row%4096];
            if (shared) value += shared[row];
            if (!isfinite(value)) atomicExch(status, 1);
            output[row] = *status ? __int_as_float(0x7fc00000) : value;
        }
        phase_finish(phase_trace,trace_epoch,layer,tokens);return;
    }
    if (draft_skip) {
        for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < 4096*tokens; row += gridDim.x * blockDim.x) {
            const int token=row/4096, neuron=row%4096;
            float active_weight=0,total_weight=0,value=0;
#pragma unroll
            for (int rank=0;rank<6;++rank) {
                const int original=token*6+rank;
                const float weight=weights[original];
                total_weight+=weight;
                if (!(state[0]&(1u<<original))) {
                    active_weight+=weight;
                    value+=weight*down[state[2+original]*4096+neuron];
                }
            }
            if (active_weight>0) value*=total_weight/active_weight;
            if (shared) value+=shared[row];
            if (!isfinite(value)) atomicExch(status,1);
            output[row]=*status?__int_as_float(0x7fc00000):value;
        }
        phase_finish(phase_trace,trace_epoch,layer,tokens);return;
    }
    __shared__ uint32_t cpu_mask;
    if (threadIdx.x == 0) {
        cpu_mask = 0;
        if (state[0] && !*status) {
            const uint64_t begin = clock_ns();
            const uint32_t ticket = state[1];
            while (!cpu_only && Word(plain(packet->response)).load(cuda::memory_order_acquire) != ticket) {
                if (clock_ns() - begin >= 1500000000ull) { atomicExch(status, 1); atomicAdd(counters + 4, 1ull); break; }
                __nanosleep(1000);
            }
            if (!*status) {
                if (cpu_only) cpu_mask = state[0];
                else if (packet->error || (packet->cpu_mask & ~state[0])) atomicExch(status, 1);
                else cpu_mask = packet->cpu_mask;
            }
            if (cpu_mask && !*status) {
                while (Word(packet->cpu_done).load(cuda::memory_order_acquire) != ticket) {
                    if (clock_ns() - begin >= 1500000000ull) { atomicExch(status, 1); atomicAdd(counters + 4, 1ull); break; }
                    __nanosleep(1000);
                }
                if (packet->cpu_error) atomicExch(status, 1);
            }
        }
        if(blockIdx.x==0)phase_stamp(phase_trace,trace_epoch,layer,tokens,2);
    }
    __syncthreads();
    if (cpu_mask && !*status) (void) Word(packet->cpu_done).load(cuda::memory_order_acquire);
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < 4096*tokens; row += gridDim.x * blockDim.x) {
        float value = 0;
        if (!*status) for (int rank = 0; rank < 6; ++rank) {
            const int original=(row/4096)*6+rank;
            const bool cpu = cpu_mask & (1u << original);
            const int compact = tokens==2 ? original : __popc(cpu_mask & ((1u << original) - 1));
            const float contribution = cpu ? packet->contribution[compact * 4096 + row%4096] : down[state[2 + original] * 4096 + row%4096];
            value += weights[original] * contribution;
        }
        if (shared) value += shared[row];
        if (!isfinite(value)) atomicExch(status, 1);
        output[row] = *status ? __int_as_float(0x7fc00000) : value;
    }
    phase_finish(phase_trace,trace_epoch,layer,tokens);
}
struct Scratch {
    NativeTgTiming timing;
    ggml_context* meta = nullptr;
    ggml_tensor *slots = nullptr, *hidden = nullptr, *down = nullptr;
    uint32_t* state = nullptr;
    void* memory = nullptr;
    uint32_t* trace_epoch=nullptr;NativeRouteRecord* trace=nullptr;
    NativePhaseClock* phase_trace=nullptr;
    unsigned long long* weight_buckets=nullptr;
    Scratch() {
        meta = ggml_init({3 * ggml_tensor_overhead(), nullptr, true});
        GGML_ASSERT(meta);
        slots = ggml_new_tensor_2d(meta, GGML_TYPE_I32, 6, 2);
        hidden = ggml_new_tensor_3d(meta, GGML_TYPE_F32, 2048, 6, 2);
        down = ggml_new_tensor_3d(meta, GGML_TYPE_F32, 4096, 6, 2);
        constexpr size_t bytes = 128 + 12 * (2048 + 4096) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&memory, bytes));
        slots->data = memory;
        state = reinterpret_cast<uint32_t*>(static_cast<char*>(memory) + 48);
        hidden->data = static_cast<char*>(memory) + 128;
        down->data = static_cast<char*>(hidden->data) + 12 * 2048 * sizeof(float);
        const char* flag=std::getenv("STRATA_DSV4_TG_ROUTE_TRACE");
        const char* weights=std::getenv("STRATA_DSV4_TG_WEIGHT_TRACE");
        if(weights&&!std::strcmp(weights,"1")){
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&weight_buckets),4*sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemset(weight_buckets,0,4*sizeof(unsigned long long)));
        }
        if((flag&&std::strcmp(flag,"1")==0)||router_prefetch_enabled()) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&trace_epoch),sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(trace_epoch,0,sizeof(uint32_t)));
            if(!flag||std::strcmp(flag,"1"))return;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&trace),8192ull*43*sizeof(NativeRouteRecord)));
            const char* clocks=std::getenv("STRATA_DSV4_TG_CLOCK_TRACE");
            if(clocks&&!std::strcmp(clocks,"1")){CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&phase_trace),8192ull*43*sizeof(NativePhaseClock)));CUDA_CHECK(cudaMemset(phase_trace,0,8192ull*43*sizeof(NativePhaseClock)));}
        }
    }
    ~Scratch() { if(weight_buckets)cudaFree(weight_buckets);if(phase_trace)cudaFree(phase_trace);if(trace)cudaFree(trace);if(trace_epoch)cudaFree(trace_epoch);if (memory) (void) cudaFree(memory); if (meta) ggml_free(meta); }
};
std::mutex scratch_mutex;
std::unordered_map<void*, std::unique_ptr<Scratch>> scratch;
}
bool native_tg_reserve(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    auto* ctx = static_cast<ggml_backend_cuda_context*>(backend->context);
    ggml_cuda_set_device(ctx->device);
    std::lock_guard<std::mutex> lock(scratch_mutex);
    if (!scratch.count(ctx)) scratch.emplace(ctx, std::make_unique<Scratch>());
    return router_prediction_reserve(backend);
}
void native_tg_release(ggml_backend_cuda_context& ctx) {
    {std::lock_guard<std::mutex> lock(scratch_mutex);scratch.erase(&ctx);}
    router_prediction_release(ctx);
}
uint32_t* native_tg_epoch_pointer(ggml_backend_cuda_context& ctx) {
    std::lock_guard<std::mutex> lock(scratch_mutex);const auto item=scratch.find(&ctx);return item==scratch.end()?nullptr:item->second->trace_epoch;
}
uintptr_t native_tg_slots(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return 0;
    std::lock_guard<std::mutex> lock(scratch_mutex);
    const auto found = scratch.find(backend->context);
    return found == scratch.end() ? 0 : reinterpret_cast<uintptr_t>(found->second->slots->data);
}
uint32_t native_tg_ready(const DeviceMissPacket* packet) {
    return Word(plain(const_cast<DeviceMissPacket*>(packet)->sequence)).load(cuda::memory_order_acquire);
}
bool native_tg_timing_begin(ggml_backend_t backend) {
    if(!backend||!ggml_backend_is_cuda(backend))return false;
    auto* ctx=static_cast<ggml_backend_cuda_context*>(backend->context);
    std::lock_guard<std::mutex> lock(scratch_mutex);
    const auto item=scratch.find(ctx);return item!=scratch.end()&&item->second->timing.begin(ctx->stream());
}
bool native_tg_timing_read(ggml_backend_t backend,NativeTgTimingResult* result) {
    if(!backend||!ggml_backend_is_cuda(backend))return false;
    auto* ctx=static_cast<ggml_backend_cuda_context*>(backend->context);
    std::lock_guard<std::mutex> lock(scratch_mutex);
    const auto item=scratch.find(ctx);return item!=scratch.end()&&item->second->timing.read(ctx->stream(),result);
}
bool native_tg_trace_read(ggml_backend_t backend,NativeRouteTrace* result,bool payload) {
    if(!backend||!ggml_backend_is_cuda(backend)||!result)return false;
    std::lock_guard<std::mutex> lock(scratch_mutex);
    const auto found=scratch.find(backend->context);if(found==scratch.end()||!found->second->trace)return false;
    auto& s=*found->second;CUDA_CHECK(cudaMemcpy(&result->epochs,s.trace_epoch,sizeof(uint32_t),cudaMemcpyDeviceToHost));
    if(s.weight_buckets)CUDA_CHECK(cudaMemcpy(result->missing_unique_weight_buckets.data(),s.weight_buckets,4*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    if(result->epochs>8192)throw std::runtime_error("native route trace fixed bound exhausted");
    if(payload){result->records.resize(size_t(result->epochs)*43);CUDA_CHECK(cudaMemcpy(result->records.data(),s.trace,result->records.size()*sizeof(NativeRouteRecord),cudaMemcpyDeviceToHost));
        if(s.phase_trace){result->phase_clocks.resize(result->records.size());CUDA_CHECK(cudaMemcpy(result->phase_clocks.data(),s.phase_trace,result->phase_clocks.size()*sizeof(NativePhaseClock),cudaMemcpyDeviceToHost));}}
    return true;
}
void native_tg_reply(DeviceMissPacket* packet, uint32_t ticket) {
    Word(plain(packet->response)).store(ticket, cuda::memory_order_release);
}
void native_tg_complete(DeviceMissPacket* packet, uint32_t ticket, bool error) {
    packet->cpu_error = error ? 1 : 0;
    Word(packet->cpu_done).store(ticket, cuda::memory_order_release);
}
void native_tg_run(ggml_backend_cuda_context& ctx, ggml_tensor* t) {
    const auto sources = native_tg_sources(t);
    Scratch* s = nullptr;
    {
        std::lock_guard<std::mutex> lock(scratch_mutex);
        auto found = scratch.find(&ctx);
        GGML_ASSERT(found != scratch.end());
        s = found->second.get();
    }
    auto* packet = reinterpret_cast<DeviceMissPacket*>(sources.packet);
    const bool split_start = (sources.reserved & 3) == 1, split_join = (sources.reserved & 3) == 2;
    const bool draft_skip = (sources.reserved & 4) != 0;
    auto* start = split_join ? t->src[0] : t;
    const int tokens=int(ggml_nelements(start->src[1])/6);
    const bool timing=(tokens==1||tokens==2)&&sources.layer<43;
    ggml_tensor slots=*s->slots,hidden=*s->hidden,down=*s->down,activation=*start->src[0];
    // Main FFN norm is [4096,T], while MMVQ IDs address [4096,1,T].
    // This is metadata only; both positions retain their contiguous values.
    activation.ne[1]=1;activation.ne[2]=tokens;activation.ne[3]=1;
    activation.nb[1]=activation.nb[2]=4096*sizeof(float);activation.nb[3]=activation.nb[2]*tokens;
    slots.ne[1]=tokens;slots.nb[2]=slots.nb[1]*tokens;
    hidden.ne[2]=down.ne[2]=tokens;hidden.nb[3]=hidden.nb[2]*tokens;down.nb[3]=down.nb[2]*tokens;
    if (!split_join) {
        if(timing)s->timing.before(sources.layer,ctx.stream());
        publish<<<1, 32, 0, ctx.stream()>>>(static_cast<const int32_t*>(start->src[6]->data),
            static_cast<const int32_t*>(start->src[1]->data), static_cast<const float*>(start->src[0]->data),
            static_cast<int32_t*>(s->slots->data), s->state, static_cast<int32_t*>(start->src[7]->data),
            static_cast<unsigned long long*>(start->src[8]->data),
            start->src[9] ? static_cast<uint32_t*>(start->src[9]->data) : nullptr,
            packet, sources.layer, sources.clamp, split_start, draft_skip, tokens, start->src[9] ? size_t(start->src[9]->ne[0]-1) : 0, s->trace_epoch, s->trace,s->phase_trace,
            static_cast<const float*>(start->src[2]->data),s->weight_buckets);
        ggml_cuda_mm_fusion_args_host fusion{};
        fusion.gate = start->src[3];
        fusion.glu_op = sources.clamp > 1e-6f ? GGML_GLU_OP_SWIGLU_CLAMP : GGML_GLU_OP_SWIGLU;
        fusion.glu_limit = sources.clamp;
        ggml_cuda_mul_mat_vec_q(ctx, start->src[4], &activation, &slots, &hidden, &fusion);
        ggml_cuda_mul_mat_vec_q(ctx, start->src[5], &hidden, &slots, &down, nullptr);
        if (split_start) { if(timing)s->timing.after(sources.layer,ctx.stream()); return; }
    } else if(timing)s->timing.before_join(sources.layer,ctx.stream());
    join<<<16, 256, 0, ctx.stream()>>>(static_cast<const float*>(s->down->data),
        static_cast<const float*>(start->src[2]->data), static_cast<float*>(t->data), s->state,
        static_cast<int32_t*>(start->src[7]->data), static_cast<unsigned long long*>(start->src[8]->data), packet,
        split_join ? static_cast<const float*>(t->src[3]->data) : nullptr, split_join, tokens,s->phase_trace,s->trace_epoch,sources.layer,draft_skip);
    CUDA_CHECK(cudaGetLastError());
    if (timing && split_join) s->timing.after_join(sources.layer,ctx.stream());
    else if(timing)s->timing.after(sources.layer,ctx.stream());
}
}
