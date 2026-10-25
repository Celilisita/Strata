#include "strata/qwen35moe/runtime.hpp"
#include "compact_expert_prefix.hpp"
#include "tg_phase_lease.hpp"
#include "head_readback.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "strata/qwen35moe/nvfp4_tg_controls.hpp"
#include "strata/qwen35moe/cpu_state.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <limits>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace strata::qwen35moe {

void Runtime::initialize_dynamic() {
    const char* flag = std::getenv("STRATA_QWEN35_TG_DYNAMIC");
    tg_dynamic_ = false;
    const bool typed=nvfp4_ && nvfp4_->device_tg;
    if(nvfp4_ && !typed)return;
    if (flag && std::string(flag) != "0" && std::string(flag) != "1")
        throw std::invalid_argument("TG_DYNAMIC must be0 or1");
    if (flag && std::string(flag) == "0") {
        if(typed)throw std::invalid_argument("NVFP4 device TG requires TG_DYNAMIC1");
        return;
    }
    int device = -1;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "dynamic device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "dynamic capabilities");
    if (!cpu_ || cpu_->from_layer != 0 || (!tg_hit_gpu_ && !typed) || !gpu_head_ ||
        !expert_cache_.slots || batch_.expert_stage.data() || resident_gather_ ||
        !properties.canMapHostMemory || !properties.unifiedAddressing) {
        if (flag || typed) throw std::invalid_argument("dynamic continuation requires shared cache, native CPU pool, GPU head and mapped UVA");
        return;
    }
    if(typed && (mtp_enabled_ || sessions_ || paired_session_ || checkpoint_sink_ ||
       verify_state_ || verify_history_ || expert_cache_.retention.enabled() ||
       expert_sources_.size()>64 || spec_.n_expert_used!=8 ||
       expert_cache_.slots<size_t(resident_layers_)*256+(expert_sources_.size()-size_t(resident_layers_))*8))
        throw std::invalid_argument("NVFP4 device TG requires ordinary single-token LRU capacity for all token routes");
    tg_dynamic_ = true;
    if(typed&&nvfp4_->tg_admission&&!expert_prefetch_)
        expert_prefetch_=std::make_unique<ExpertPrefetch>(expert_sources_.size());
    std::vector<core::NativeMissSource> sources;
    for (size_t il=0;il<expert_sources_.size();++il) {
        const auto& layer=expert_sources_[il];
        core::NativeMissSource source;
        source.gate = gguf_.tensor_data(*layer.gate_exps);
        source.up = gguf_.tensor_data(*layer.up_exps);
        source.down = gguf_.tensor_data(*layer.down_exps);
        source.n_expert = int(spec_.n_expert);
        if(typed) {
            source.gate_scales=nvfp4_->tg_cpu_scales[0].data()+il*256;
            source.up_scales=nvfp4_->tg_cpu_scales[1].data()+il*256;
            source.down_scales=nvfp4_->tg_cpu_scales[2].data()+il*256;
        }
        std::string error;
        if (!kernels::cpu::native_fmt(int(layer.gate_exps->type), int(layer.down_exps->type),
                                       int(spec_.n_embd), int(spec_.n_ff_exp), source.format, error))
            throw std::runtime_error(error);
        sources.push_back(source);
    }
    // Only a delivery adapter: the pool, weight views and cache remain authoritative.
    miss_service_ = std::make_unique<core::NativeMissService>(
        cpu_->pool, std::move(sources), int(spec_.n_embd), int(spec_.n_expert_used), (mtp_enabled_ || sessions_) ? 2 : 1,typed);
    expert_cache_.resident_slots.reset(expert_cache_.key_to_slot.size() * sizeof(int32_t));
    dynamic_args_.reset(expert_sources_.size() * sizeof(core::NativeMissArgs));
    if (mtp_enabled_ || sessions_) dynamic_pair_args_.reset(dynamic_args_.bytes());
    dynamic_frame_.reset(sizeof(core::NativeMissFrame));
    dynamic_ages_.reset(expert_cache_.age.size() * sizeof(uint64_t));
}

void Runtime::start_decode(Metrics* metrics) {
    const bool phase_lease = detail::tg_phase_lease_enabled();
    const bool phase_verify = detail::tg_phase_verify_enabled();
    if (phase_verify && !phase_lease)
        throw std::invalid_argument("TG_PHASE_VERIFY requires TG_PHASE_LEASE=1");
    if (phase_lease && (!tg_dynamic_ || mtp_enabled_ || session_capacity() != 1 ||
        decode_phase_ || checkpoint_sink_ || mtp_sink_ || !batch_.capacity ||
        batch_.expert_stage.data() || !expert_cache_.slots || expert_cache_.retention.leased()))
        throw std::invalid_argument("TG phase lease requires a prepared idle single-session dynamic boundary, MTPoff and no sink");
    detail::verify_compact_prefix(expert_cache_,expert_sources_);
    prefetch_drain();
    if (metrics) collect_visual_metrics(*metrics);
    if (!tg_dynamic_) { decode_phase_ = true; return; }
    if(nvfp4_ && nvfp4_->device_tg && expert_cache_.slots<size_t(resident_layers_)*256+
       (expert_sources_.size()-size_t(resident_layers_))*8)
        throw std::runtime_error("NVFP4 device TG cache shrank below token-replay capacity");
    cuda_check(cudaStreamSynchronize(stream_), "drain before decode epoch");
    miss_service_->stop();
    if (phase_lease) {
        auto& b = batch_; auto& cache = expert_cache_;
        Metrics local; Metrics& work = metrics ? *metrics : local;
        const size_t old_slots = cache.slots, old_batch_bytes = b.owned_bytes();
        const int old_capacity = b.capacity;
        std::vector<uint8_t*> old_addresses(old_slots);
        for (size_t i = 0; i < old_slots; ++i) old_addresses[i] = cache.data.slot(i);
        const auto old_keys = cache.slot_to_key; const auto old_ages = cache.age;
        const uint64_t before_bytes = work.expert_h2d, before_calls = work.expert_h2d_calls;
        batch_failed_ = true;
        try {
            invalidate_moe_graphs();
            // Providers own only PP control/workspace, never expert identity.
            projection_mmq_.reset(); b.down_mmq.reset(); b.routed_tc.reset();
            allocate_batch(16); // preserve Gemm, CPU pool, KV, GDN and work_ state
            prefill_budget_capacity_ = 16; // next reserve_context must return the loan
            const size_t slots = expert_cache_budget(cache.data.bytes(), 0, kv_capacity_, 16);
            if (slots < old_slots)
                throw std::runtime_error("TG phase lease cannot preserve existing cache under current budget");
            if (slots > old_slots) resize_expert_cache(slots, work);
            // Even if old holes exist, fill only newly allocated tail slots.
            cache.next = old_slots;
            size_t filled = 0;
            for (size_t key = 0; key < cache.key_to_slot.size() && filled < slots-old_slots; ++key) {
                if (cache.key_to_slot[key] >= 0) continue;
                if (cache.next < old_slots || cache.next >= slots || cache.slot_to_key[cache.next] >= 0)
                    throw std::logic_error("TG lease tail destination is not empty");
                if (cache.data.compact()) {
                    const size_t gu=kernels::native_mmvq_weight_bytes(12,2048,512);
                    const size_t payload=2*gu+kernels::native_mmvq_weight_bytes(int(expert_sources_[key/256].down_exps->type),512,2048);
                    if (payload!=cache.data.capacity(cache.next))
                        throw std::logic_error("TG tail is not a compatible missing Q4 key");
                }
                expert_view(key/256, int(key%256), expert_sources_[key/256], work);
                const int slot = cache.key_to_slot[key];
                if (slot < 0 || size_t(slot) < old_slots || cache.slot_to_key[size_t(slot)] != int(key))
                    throw std::logic_error("TG lease key inverse mismatch");
                ++filled;
            }
            if (filled != slots-old_slots) throw std::logic_error("TG lease inventory cannot fill tail");
            for (size_t i = 0; i < old_slots; ++i) {
                if (cache.slot_to_key[i] != old_keys[i] || cache.age[i] != old_ages[i] ||
                    cache.data.slot(i) != old_addresses[i] ||
                    (old_keys[i] >= 0 && cache.key_to_slot[size_t(old_keys[i])] != int(i)))
                    throw std::logic_error("TG phase lease changed existing residency");
            }
            cache.next = 0;
            while (cache.next < slots && cache.slot_to_key[cache.next] >= 0) ++cache.next;
            prepare_moe_graphs(); // capture only; publishes the replacement batch views
            if (moe_graph_) ++work.explicit_syncs; // existing graph preparation drain
            if (phase_verify) {
                const auto begin = std::chrono::steady_clock::now();
                size_t verified_keys = 0, verified_bytes = 0, d2h_bytes = 0, sync_calls = 0;
                size_t types[3] = {}; uint64_t descriptor = 14695981039346656037ULL;
                std::vector<uint8_t> readback; // one bounded buffer; dead before epoch starts
                auto receipt = [&](int failures) {
                    const double elapsed = std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-begin).count();
                    std::fprintf(stderr, "qwen35_tg_phase_verify new_keys=%zu verified_keys=%zu verified_bytes=%zu d2h_bytes=%zu extra_sync_calls=%zu host_buffer_bytes=%zu down_q4=%zu down_q5=%zu down_q6=%zu descriptor_fnv64=%llu failures=%d verify_ms=%.6f diagnostic_only=1\n",
                        slots-old_slots, verified_keys, verified_bytes, d2h_bytes, sync_calls,
                        readback.size(), types[0], types[1], types[2],
                        static_cast<unsigned long long>(descriptor), failures, elapsed);
                    std::fflush(stderr);
                };
                auto bind = [&](uint64_t value) {
                    for (int i=0;i<8;++i) { descriptor ^= uint8_t(value>>(8*i)); descriptor *= 1099511628211ULL; }
                };
                try {
                    size_t maximum_payload = 0;
                    for (const auto& source : expert_sources_) {
                        if (!source.down_exps) throw std::logic_error("TG verify missing Down metadata");
                        const int type = int(source.down_exps->type);
                        if (type!=12 && type!=13 && type!=14)
                            throw std::logic_error("TG verify unsupported Down format");
                        const size_t gu = kernels::native_mmvq_weight_bytes(12,2048,512);
                        const size_t down = kernels::native_mmvq_weight_bytes(type,512,2048);
                        if (2*gu+down > maximum_payload) maximum_payload = 2*gu+down;
                    }
                    if (!maximum_payload || maximum_payload > cache.stride || maximum_payload > (size_t(2)<<20))
                        throw std::logic_error("TG tail verify buffer exceeds real expert/cache bounds");
                    readback.resize(maximum_payload);
                    // Graph preparation already drains this stream. With graphs off,
                    // one diagnostic boundary drain publishes all newly filled bytes.
                    if (!moe_graph_) {
                        ++sync_calls; ++work.explicit_syncs;
                        cuda_check(cudaStreamSynchronize(stream_), "publish TG verify tail");
                    }
                    if (gguf_.data_start() > gguf_.file_size())
                        throw std::logic_error("TG verify invalid GGUF data extent");
                    const uint64_t available = gguf_.file_size()-gguf_.data_start();
                    const int inputs[3] = {2048,2048,512}, outputs[3] = {512,512,2048};
                    const char* roles[3] = {"gate","up","down"};
                    for (size_t slot=old_slots;slot<slots;++slot) {
                        const int key = cache.slot_to_key[slot];
                        if (key < 0 || size_t(key) >= cache.key_to_slot.size() ||
                            cache.key_to_slot[size_t(key)] != int(slot))
                            throw std::logic_error("TG verify directory inverse mismatch");
                        const size_t layer = size_t(key)/256, id = size_t(key)%256;
                        if (layer >= expert_sources_.size()) throw std::logic_error("TG verify source layer");
                        const auto& source = expert_sources_[layer];
                        const TensorInfo* tensors[3] = {source.gate_exps,source.up_exps,source.down_exps};
                        size_t bytes[3] = {}; const uint8_t* original[3] = {};
                        bind(uint64_t(key)); bind(slot);
                        for (int role=0;role<3;++role) {
                            const auto* tensor = tensors[role];
                            const int type = tensor ? int(tensor->type) : -1;
                            if (!tensor || (role<2 ? type!=12 : (type!=12 && type!=13 && type!=14)) ||
                                tensor->name != "blk."+std::to_string(layer)+".ffn_"+roles[role]+"_exps.weight" ||
                                tensor->shape.size()!=3 || tensor->shape[0]!=uint64_t(inputs[role]) ||
                                tensor->shape[1]!=uint64_t(outputs[role]) || tensor->shape[2]!=256)
                                throw std::logic_error("TG verify tensor role/format/shape mismatch");
                            bytes[role] = kernels::native_mmvq_weight_bytes(type,inputs[role],outputs[role]);
                            if (tensor->offset > available || bytes[role] > (available-tensor->offset)/256)
                                throw std::logic_error("TG verify GGUF expert extent mismatch");
                            const uint64_t offset = tensor->offset + id*bytes[role];
                            original[role] = gguf_.tensor_data(*tensor)+id*bytes[role];
                            bind(uint64_t(role)); bind(uint64_t(type)); bind(gguf_.data_start()+offset); bind(bytes[role]);
                        }
                        const size_t payload = bytes[0]+bytes[1]+bytes[2];
                        if (cache.data.compact() && payload!=cache.data.capacity(slot))
                            throw std::logic_error("TG verified key capacity mismatch");
                        if (bytes[0]!=bytes[1] || payload > readback.size())
                            throw std::logic_error("TG verify cache role bounds mismatch");
                        cuda_check(cudaMemcpyAsync(readback.data(),cache.data.slot(slot),payload,
                            cudaMemcpyDeviceToHost,stream_), "read real TG tail payload");
                        ++sync_calls; ++work.explicit_syncs;
                        cuda_check(cudaStreamSynchronize(stream_), "wait TG tail payload readback");
                        d2h_bytes += payload;
                        size_t first = 0;
                        for (int role=0;role<3;++role) {
                            if (std::memcmp(readback.data()+first,original[role],bytes[role]) != 0)
                                throw std::runtime_error("TG tail payload mismatch key="+std::to_string(key)+" role="+roles[role]);
                            first += bytes[role];
                        }
                        ++verified_keys; verified_bytes += payload;
                        ++types[int(source.down_exps->type)-12];
                    }
                    if (verified_keys != slots-old_slots)
                        throw std::logic_error("TG tail verification coverage mismatch");
                } catch (...) {
                    // Keep the host destination alive until any submitted readback drains.
                    ++sync_calls; ++work.explicit_syncs;
                    (void)cudaStreamSynchronize(stream_);
                    receipt(1); throw;
                }
                receipt(0);
            }
            batch_failed_ = false;
            std::fprintf(stderr, "qwen35_tg_phase_lease batch=%d->16 owned_batch=%zu->%zu slots=%zu->%zu stride=%zu new_keys=%zu fill_bytes=%llu fill_calls=%llu retained_exact=1 reserve_mib=128 compact=%d backing_bytes=%zu high_slots=%zu\n",
                old_capacity, old_batch_bytes, b.owned_bytes(), old_slots, slots, cache.stride, filled,
                static_cast<unsigned long long>(work.expert_h2d-before_bytes),
                static_cast<unsigned long long>(work.expert_h2d_calls-before_calls),
                int(cache.data.compact()),cache.data.bytes(),cache.data.high_slots());
            std::fflush(stderr);
        } catch (...) { batch_failed_ = true; throw; }
    }
    if (decode_epoch_ == (std::numeric_limits<uint32_t>::max)())
        throw std::runtime_error("decode epoch exhausted; recreate runtime");
    core::NativeMissFrame frame{};
    frame.epoch = ++decode_epoch_;
    frame.cache_tick = expert_cache_.tick;
    if(nvfp4_ && nvfp4_->device_tg){nvfp4_->tg_token=0;nvfp4_->tg_pending=false;
        nvfp4_->tg_directory=expert_cache_.key_to_slot;nvfp4_->tg_tail_bytes=nvfp4_->tg_tail_control=0;}
    std::vector<core::NativeMissArgs> args(expert_sources_.size());
    int device = -1;
    int clock_khz = 0;
    cuda_check(cudaGetDevice(&device), "decode device");
    cuda_check(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device), "decode clock");
    if (clock_khz <= 0) throw std::runtime_error("invalid decode clock rate");
    unsigned timeout_ms = 50;
    if (const char* raw = std::getenv("STRATA_QWEN35_MISS_TIMEOUT_MS")) {
        const std::string value(raw);
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("invalid dynamic miss timeout");
        const auto parsed = std::stoul(value);
        if (parsed < 1 || parsed > 1000)
            throw std::invalid_argument("dynamic miss timeout must be 1..1000 ms");
        timeout_ms = unsigned(parsed);
    }
    for (size_t il = 0; il < args.size(); ++il) {
        auto& a = args[il];
        a = miss_service_->device_args();
        a.frame = dynamic_frame_.as<core::NativeMissFrame>();
        a.slots = expert_cache_.resident_slots.as<int32_t>() + il * spec_.n_expert;
        a.cache_slot_addresses = expert_cache_.data.device_addresses();
        a.cache_stride = expert_cache_.stride;
        a.ages = dynamic_ages_.as<uint64_t>();
        a.gu_bytes = nvfp4_ ? Nvfp4ExpertInventory::matrix_bytes :
            kernels::native_mmvq_weight_bytes(12, int(spec_.n_embd), int(spec_.n_ff_exp));
        a.norm = batch_.norm.as<float>();
        a.gu_addresses = batch_.expert_addresses.as<unsigned long long>();
        a.down_addresses = batch_.down_addresses.as<unsigned long long>();
        a.gu_rows = batch_.expert_rows.as<int32_t>();
        a.down_rows = batch_.down_rows.as<int32_t>();
        a.layer = int(il); a.n_expert = int(spec_.n_expert);
        if(nvfp4_ && nvfp4_->device_tg)
            bind_nvfp4_tg(a,batch_.expert_q8.data(),batch_.expert_q8.bytes(),nvfp4_->tg_globals.as<float>());
        a.timeout_cycles = uint64_t(clock_khz) * timeout_ms; // kHz * ms
    }
    std::vector<core::NativeMissArgs> pair_args;
    if (dynamic_pair_args_.bytes()) {
        pair_args = args;
        for (auto& a : pair_args) { a.tokens = 2; a.ranks *= 2; }
    }
    cuda_check(cudaMemcpyAsync(dynamic_frame_.data(), &frame, sizeof(frame),
                               cudaMemcpyHostToDevice, stream_), "decode frame");
    cuda_check(cudaMemcpyAsync(expert_cache_.resident_slots.data(), expert_cache_.key_to_slot.data(),
                               expert_cache_.resident_slots.bytes(), cudaMemcpyHostToDevice, stream_),
               "decode directory snapshot");
    cuda_check(cudaMemcpyAsync(dynamic_ages_.data(), expert_cache_.age.data(), dynamic_ages_.bytes(),
                               cudaMemcpyHostToDevice, stream_), "decode cache ages");
    cuda_check(cudaMemcpyAsync(dynamic_args_.data(), args.data(), dynamic_args_.bytes(),
                               cudaMemcpyHostToDevice, stream_), "decode argument views");
    if (dynamic_pair_args_.bytes()) {
        cuda_check(cudaMemcpyAsync(dynamic_pair_args_.data(), pair_args.data(), dynamic_pair_args_.bytes(),
            cudaMemcpyHostToDevice, stream_), "decode pair arguments");
    }
    // Local vectors may be reused only after their upload finishes.
    cuda_check(cudaStreamSynchronize(stream_), "finish decode argument upload");
    if (head_readback_) head_readback_->enter_decode();
    if(nvfp4_&&nvfp4_->tg_admission)
        nvfp4_->tg_admission->open(*expert_prefetch_,prefetch_cache(),host_readonly_,frame.epoch,
            expert_cache_.resident_slots.as<int32_t>(),expert_cache_.resident_slots.bytes());
    miss_service_->start(frame.epoch);
    miss_accounted_ = {}; dynamic_layers_ = dynamic_accounted_layers_ = 0;
    dynamic_entries_ = dynamic_accounted_entries_ = 0;
    if (metrics) {
        metrics->control_h2d += dynamic_frame_.bytes() + dynamic_args_.bytes() +
                                expert_cache_.resident_slots.bytes() + dynamic_ages_.bytes() + dynamic_pair_args_.bytes();
        metrics->explicit_syncs += 2;
    }
    decode_phase_ = true;
}

// Ordinary dynamic epochs merge device LRU. Typed epochs replay it at each token
// head boundary; finish still drains final queued admissions before reset/free.
void Runtime::finish_decode(Metrics* metrics) {
    if (metrics) collect_visual_metrics(*metrics);
    if (!tg_dynamic_ || !decode_phase_) return;
    const auto submit_begin=std::chrono::steady_clock::now();
    if(nvfp4_&&nvfp4_->tg_admission){Metrics tail;flush_nvfp4_admission(metrics?*metrics:tail);}
    const double submit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-submit_begin).count();
    core::NativeMissFrame frame{};
    cuda_check(cudaMemcpyAsync(&frame, dynamic_frame_.data(), sizeof(frame),
                               cudaMemcpyDeviceToHost, stream_), "finish decode frame");
    const bool typed=nvfp4_ && nvfp4_->device_tg;
    if(!typed)cuda_check(cudaMemcpyAsync(expert_cache_.age.data(), dynamic_ages_.data(), dynamic_ages_.bytes(),
                               cudaMemcpyDeviceToHost, stream_), "finish decode cache ages");
    const auto drain_start=std::chrono::steady_clock::now();
    const auto drained=cudaStreamSynchronize(stream_);
    if(drained!=cudaSuccess&&typed)batch_failed_=true;
    cuda_check(drained, "finish decode policy merge");
    if (head_readback_) head_readback_->leave_decode();
    // Charge the request-boundary policy restore to the existing tail wall bound.
    const double drain_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-drain_start).count();
    miss_service_->stop();
    if(nvfp4_&&nvfp4_->tg_admission&&nvfp4_->tg_admission->opened()){
        try{nvfp4_->tg_admission->close();}
        catch(...){nvfp4_->tg_admission->invalidate();batch_failed_=true;throw;}
    }
    if (frame.failed) throw std::runtime_error("failed device decode epoch");
    if(!typed)expert_cache_.tick = frame.cache_tick;
    else {
        float tail_ms=0;
        if(nvfp4_->tg_pending)cuda_check(cudaEventElapsedTime(&tail_ms,nvfp4_->tg_admit_begin,nvfp4_->tg_admit_end),"NVFP4 tail admission cost");
        std::fprintf(stderr,"NVFP4_TG_TAIL {\"epoch\":%u,\"tg_token\":%u,\"expert_h2d\":%llu,\"control_h2d\":%llu,\"admission_gpu_ms\":%.9g,\"drain_wall_ms\":%.9g,\"admission_submit_wall_ms\":%.9g,\"scope\":\"%s\"}\n",
            decode_epoch_,nvfp4_->tg_token,static_cast<unsigned long long>(nvfp4_->tg_tail_bytes),
            static_cast<unsigned long long>(nvfp4_->tg_tail_control),double(tail_ms),drain_ms,submit_ms,
            nvfp4_->tg_admission?"last_reserved_admission_flush":"last_post_head_admission");
        std::fflush(stderr);nvfp4_->tg_pending=false;
    }
    if (metrics) {
        collect_dynamic_metrics(*metrics);
        metrics->control_d2h += sizeof(frame) + (typed?0:dynamic_ages_.bytes());
        ++metrics->explicit_syncs;
    }
    decode_phase_ = false;
}

void Runtime::batch_moe_device_eager(size_t il, const LayerWeights& layer, Metrics& metrics, int count) {
    if (count < 1 || count > 2 || (count == 2 && !dynamic_pair_args_.bytes()))
        throw std::invalid_argument("unprepared dynamic pair");
    if(nvfp4_){if(count!=1)throw std::invalid_argument("NVFP4 TG token count");
        batch_moe_nvfp4_device(il,layer,metrics);return;}
    auto& b = batch_;
    route_project(layer.router.f32(), b.norm.as<float>(), b.logits.as<float>(),
                   int(spec_.n_embd), int(spec_.n_expert) + 1, stream_, count);
    auto& args = count == 2 ? dynamic_pair_args_ : dynamic_args_;
    route_top8_continue(b.logits.as<float>(), b.ids.as<int32_t>(), b.weights.as<float>(),
                        args.as<core::NativeMissArgs>() + il, stream_, count);
    batch_moe_grouped_compute(il, layer, count, "", metrics);
    ++dynamic_layers_; dynamic_entries_ += size_t(count) * spec_.n_expert_used;
}

// Called after the existing head synchronization. No device readback is added.
void Runtime::collect_dynamic_metrics(Metrics& metrics) {
    const auto current = miss_service_->snapshot();
    const auto requests = current.requests - miss_accounted_.requests;
    const auto experts = current.experts - miss_accounted_.experts;
    const auto layers = dynamic_layers_ - dynamic_accounted_layers_;
    metrics.route_d2h += current.id_bytes - miss_accounted_.id_bytes;
    metrics.cpu_norm_d2h += current.norm_bytes - miss_accounted_.norm_bytes;
    metrics.cpu_result_h2d += current.result_bytes - miss_accounted_.result_bytes;
    metrics.cpu_expert_read += current.expert_bytes - miss_accounted_.expert_bytes;
    metrics.cpu_pool_ms += current.pool_ms - miss_accounted_.pool_ms;
    metrics.cpu_gu_ms += current.gu_ms - miss_accounted_.gu_ms;
    metrics.cpu_quant_ms += current.quant_ms - miss_accounted_.quant_ms;
    metrics.cpu_down_ms += current.down_ms - miss_accounted_.down_ms;
    metrics.cpu_jobs += current.jobs - miss_accounted_.jobs;
    metrics.cpu_paired_jobs += current.paired_jobs - miss_accounted_.paired_jobs;
    metrics.host_continuations += requests;
    metrics.tg_cpu_layers += requests;
    metrics.tg_gpu_hit_layers += layers - requests;
    metrics.tg_device_layers += layers - requests;
    metrics.expert_cache_hits += dynamic_entries_ - dynamic_accounted_entries_ - experts;
    if(nvfp4_)metrics.expert_cache_misses+=experts;
    miss_accounted_ = current; dynamic_accounted_layers_ = dynamic_layers_;
    dynamic_accounted_entries_ = dynamic_entries_;
}

} // namespace strata::qwen35moe
