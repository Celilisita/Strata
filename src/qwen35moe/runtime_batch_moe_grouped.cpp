#include "strata/qwen35moe/runtime.hpp"
#include "tg_gu_cluster.hpp"
#include "tg_quant_down_pair.hpp"
#include "strata/qwen35moe/shared_f16_gemm.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_quantize_rows.hpp"
#include "strata/qwen35moe/mapped_pp.hpp"
#ifdef STRATA_BLACKWELL_MMQ
#include "strata/prefill/native_blackwell_mmq.hpp"
#endif
#include <cstdlib>
#include "strata/prefill/native_code_sum_support.hpp"
#include "../prefill/native_down_direct_packed.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace strata::qwen35moe {

bool tg_gu_cluster_enabled(){
    static const bool enabled=tg_gu_cluster_flag(std::getenv("STRATA_QWEN35_TG_GU_CLUSTER"));
    return enabled;
}
bool tg_gu_cluster_provider(bool request,const TgGuClusterView& view,void* stream){
    const auto& a=view.native;
    if(!stream||!a.addresses||!a.input_rows||!a.down_addresses||!a.input_q8||
       !a.shared_gate||!a.shared_up||!a.expert_hidden||!a.shared_hidden||
       !view.expert_gu||!view.shared_gate_output||!view.shared_up_output)
        throw std::invalid_argument("invalid TG GU cluster view");
    if(request)return kernels::native_tg_gu_cluster(a,stream);
    kernels::native_mmvq_grouped(12,a.addresses,a.input_rows,a.input_q8,view.expert_gu,2048,512,16,stream);
    swiglu(view.expert_gu,view.expert_gu+8*512,a.expert_hidden,8*512,stream,a.down_addresses,512);
    kernels::native_mmvq(12,a.shared_gate,a.input_q8,view.shared_gate_output,2048,512,1,stream);
    kernels::native_mmvq(12,a.shared_up,a.input_q8,view.shared_up_output,2048,512,1,stream);
    swiglu(view.shared_gate_output,view.shared_up_output,a.shared_hidden,512,stream);
    return false;
}

bool tg_quant_down_pair_enabled(){
    static const bool enabled=[] {
        const bool pair=tg_quant_down_pair_flag(std::getenv("STRATA_QWEN35_TG_QUANT_DOWN_PAIR"));
        if(!pair)return false;
        return tg_quant_down_pair_dependencies(pair,tg_gu_cluster_enabled(),
            tg_quant_down_pair_flag(std::getenv("STRATA_QWEN35_TG_DOWN_ROW4")));
    }();return enabled;
}
bool tg_quant_down_pair_provider(bool request,int type,const kernels::NativeTgQuantDownPair& a,void* stream){
    if(request)return kernels::native_tg_quant_down_pair(type,a,stream);
    kernels::native_quantize_rows(a.expert_hidden,a.expert_q8,512,8,false,stream,a.down_addresses);
    kernels::native_mmvq_grouped(type,a.down_addresses,a.down_rows,a.expert_q8,a.expert_parts,512,2048,8,stream);
    kernels::native_quantize_rows(a.shared_hidden,a.shared_q8,512,1,false,stream);
    kernels::native_mmvq(type,a.shared_down,a.shared_q8,a.shared_output,512,2048,1,stream);
    return false;
}

void Runtime::batch_moe_grouped(size_t il, const LayerWeights& layer, int count,
                                const std::string& reference, Metrics& metrics) {
    const auto ids = batch_route(layer, count, reference, metrics, false);
    batch_moe_grouped_ready(il, layer, count, reference, metrics, ids);
}

void Runtime::batch_moe_grouped_ready(size_t il, const LayerWeights& layer, int count,
                                      const std::string& reference, Metrics& metrics,
                                      const std::vector<int32_t>& ids) {
    const bool measure_table = decode_phase_ && tg_diag_ && count == 1 && cpu_;
    std::chrono::steady_clock::time_point table_start;
    if (measure_table) table_start = std::chrono::steady_clock::now();
    auto& b = batch_;
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    const int entries = count * 8;
    if (ids.size() != size_t(entries))
        throw std::invalid_argument("grouped route count mismatch");

    // Frequency retention changes eviction only. The current layer is leased
    // until all grouped work is submitted on the existing ordered stream.
    // One layer's entire inventory suffices; a second layer is not required.
    // Existing staging and smaller protected-prefix budgets keep their schedule.
    const bool retain = !decode_phase_ && !b.expert_stage.data() &&
        expert_cache_.slots >= size_t(resident_layers_ + 1) * 256;
    core::ExpertRetention::LayerLease retention(expert_cache_.retention, il,
        expert_sources_.size(), 256, ids, retain);

    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][2], stream_), "prefill copy begin");
    prefetch_routes(il, ids); // Exposed copy wait is inside existing copy interval.
    const auto newly_loaded=prepare_expert_copy_runs(il,count,reference,ids,metrics);
    const auto gu_reserved=!decode_phase_&&prefill_config_.gu_copy_wave ? prepare_gu_copy(il,count,reference,ids,metrics) : 0;
    const bool profile_address = !decode_phase_ && !prefill_events_.empty();
    const auto address_start = profile_address ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& gu_address = b.host_expert_addresses;
    auto& gu_row = b.host_expert_rows;
    auto& down_address = b.host_down_addresses;
    auto& down_row = b.host_down_rows;
    std::array<bool, 256> requested{};
    if (b.expert_stage.data()) {
        if (b.expert_stage_layer != il) {
            b.expert_stage_keys.fill(-1);
            std::fill(b.expert_stage_slot_keys.begin(), b.expert_stage_slot_keys.end(), -1);
            std::fill(b.expert_stage_age.begin(), b.expert_stage_age.end(), 0);
            b.expert_stage_next = 0;
            b.expert_stage_layer = il;
        }
        for (int id : ids) {
            if (id < 0 || id >= 256) throw std::runtime_error("invalid staged expert ID");
            requested[size_t(id)] = true;
        }
    }
    const size_t gu_bytes = kernels::native_mmvq_weight_bytes(12, 2048, 512);
    const int down_type = int(expert_sources_[il].down_exps->type);
    std::optional<MappedPpLayer> mapped;
#ifdef STRATA_BLACKWELL_MMQ
    // SM120 source definition is required for THIS translation unit too.
    // Enough nonresident slots preserve all real/fallback views until compute.
    if (prefill_config_.mapped_pp && !decode_phase_ && count >= 1024 && count <= 2048 &&
        !b.expert_stage.data() && b.down_mmq && prefill_config_.grouped_gu_mmq && prefill_config_.grouped_down_mmq &&
        spec_.n_embd == 2048 && spec_.n_ff_exp == 512 && spec_.n_expert == 256 &&
        int(expert_sources_[il].gate_exps->type) == 12 && int(expert_sources_[il].up_exps->type) == 12 &&
        (down_type == 12 || down_type == 13) &&
        expert_cache_.slots >= size_t(resident_layers_ + 1) * 256 &&
        expert_cache_.key_to_slot.size() >= (il + 1) * 256 && host_readonly_.bytes() &&
        prefill::mmq::blackwell_mmq_backend(12, 2) == 6 && prefill::mmq::blackwell_mmq_backend(down_type, 1) == 6) {
        const char* setting = std::getenv("STRATA_BENCH_GROUP_ROWS");
        if (setting && std::strcmp(setting, "64") && std::strcmp(setting, "128") && std::strcmp(setting, "auto"))
            throw std::invalid_argument("mapped PP group rows require64,128 orauto");
        // The auto provider keeps64 for this path's entries<=16384.
        const int tile_rows = setting && !std::strcmp(setting, "128") ? 128 : 64;
        const auto& source = expert_sources_[il];
        mapped.emplace(host_readonly_, gguf_.tensor_data(*source.gate_exps), gguf_.tensor_data(*source.up_exps),
            gguf_.tensor_data(*source.down_exps), gu_bytes,
            kernels::native_mmvq_weight_bytes(down_type, 512, 2048), tile_rows, ids.data(), ids.size());
    }
#endif
    for (int entry = 0; entry < entries; ++entry) {
        const int token = entry / 8;
        const int id = ids[size_t(entry)];
        if (id < 0 || id >= 256) throw std::runtime_error("invalid grouped expert ID");
        ExpertView view;
        if (b.expert_stage.data()) {
            int slot = b.expert_stage_keys[size_t(id)];
            if (slot < 0) {
                if (b.expert_stage_next < b.expert_stage_slots) {
                    slot = int(b.expert_stage_next++);
                } else {
                    uint64_t oldest = uint64_t(-1);
                    for (size_t s = 0; s < b.expert_stage_slots; ++s) {
                        const int old_id = b.expert_stage_slot_keys[s];
                        if (old_id >= 0 && !requested[size_t(old_id)] &&
                            b.expert_stage_age[s] < oldest) {
                            oldest = b.expert_stage_age[s];
                            slot = int(s);
                        }
                    }
                    if (slot < 0) throw std::runtime_error("no safe Qwen35 staging victim");
                }
                const int old_id = b.expert_stage_slot_keys[size_t(slot)];
                if (old_id >= 0) b.expert_stage_keys[size_t(old_id)] = -1;
                b.expert_stage_keys[size_t(id)] = slot;
                b.expert_stage_slot_keys[size_t(slot)] = id;
                view = staged_expert_view(il, id, slot, expert_sources_[il], metrics);
            } else {
                ++metrics.expert_cache_hits;
                auto* address = b.expert_stage.as<uint8_t>() +
                                size_t(slot) * b.expert_stage_stride;
                view = {address, address + gu_bytes, address + 2 * gu_bytes};
            }
            b.expert_stage_age[size_t(slot)] = ++b.expert_stage_tick;
        } else {
            MappedPpWeights alias;
            if (mapped && mapped->view(id, expert_cache_.key_to_slot[il * 256 + size_t(id)] >= 0, alias))
                view = {alias.gate, alias.up, alias.down};
            else
                view = expert_view(il, id, expert_sources_[il], metrics);
        }
        gu_address[size_t(entry)] =
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(view.gate));
        gu_address[size_t(entries + entry)] =
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(view.up));
        gu_row[size_t(entry)] = token;
        gu_row[size_t(entries + entry)] = token;
        down_address[size_t(entry)] =
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(view.down));
        down_row[size_t(entry)] = entry;
    }
    if(metrics.expert_cache_hits<newly_loaded+gu_reserved)throw std::runtime_error("copy-runs hit accounting underflow");
    metrics.expert_cache_hits-=newly_loaded+gu_reserved; // First real cold reference remains a miss.
    if (mapped) {
        const auto cost = mapped->cost();
        metrics.pp_mapped_entries += cost.entries;
        metrics.pp_mapped_unique += cost.unique;
        metrics.pp_mapped_logical_weight_bytes += cost.logical_weight_bytes;
        metrics.pp_mapped_expected_tile_bytes += cost.expected_tile_bytes;
    }
    if (profile_address) metrics.pp_address_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - address_start).count();
    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][3], stream_), "prefill copy end");
    std::chrono::steady_clock::time_point table_copy_start;
    if (measure_table) table_copy_start = std::chrono::steady_clock::now();
    const bool gathered_gu = (!decode_phase_ && count >= 1024 && prefill_config_.grouped_gu_mmq &&
        prefill::mmq::native_code_sum_grouped_supported(12, 2048, 512)) ||
                             (b.routed_tc && routed_tc_enabled_ && routed_gu_gathered_ &&
                             !decode_phase_ && count >= 1024);
    if (!gathered_gu) {
        cuda_check(cudaMemcpyAsync(b.expert_addresses.data(), gu_address.data(),
            size_t(entries * 2) * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream_),
            "grouped gate/up addresses");
        cuda_check(cudaMemcpyAsync(b.expert_rows.data(), gu_row.data(),
            size_t(entries * 2) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_),
            "grouped gate/up rows");
        metrics.control_h2d += size_t(entries * 2) * (sizeof(unsigned long long) + sizeof(int32_t));
    }
    const bool tc_down = (down_type == 12 || down_type == 14) && b.routed_tc && routed_tc_enabled_ &&
        !decode_phase_ && count >= 1024;
    bool q5_mmq_down = false;
#ifdef STRATA_BLACKWELL_MMQ
    // Match batch_moe_down_mmq: Q5 PP consumes host tables through its own
    // grouped plan; the legacy device tables are needed only by the fallback.
    q5_mmq_down = down_type == 13 && !decode_phase_ && count >= 1024 &&
        prefill_config_.grouped_down_mmq && prefill::mmq::blackwell_mmq_backend(down_type, 1) == 6;
#endif
    if (!tc_down && !q5_mmq_down) {
        cuda_check(cudaMemcpyAsync(b.down_addresses.data(), down_address.data(),
            size_t(entries) * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream_),
            "grouped down addresses");
        cuda_check(cudaMemcpyAsync(b.down_rows.data(), down_row.data(),
            size_t(entries) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_),
            "grouped down rows");
        metrics.control_h2d += size_t(entries) * (sizeof(unsigned long long) + sizeof(int32_t));
    }
    if (measure_table) {
        const auto done = std::chrono::steady_clock::now();
        metrics.diag_hit_table_ms += std::chrono::duration<double, std::milli>(
            done - table_start).count();
        metrics.diag_hit_table_copy_ms += std::chrono::duration<double, std::milli>(
            done - table_copy_start).count();
    }

    batch_moe_grouped_compute(il, layer, count, reference, metrics);
}

void Runtime::batch_moe_resident(size_t il, const LayerWeights& layer, Metrics& metrics) {
    auto& b = batch_;
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    route_project(layer.router.f32(), f(b.norm), f(b.logits), 2048, 257, stream_);
    route_top8(f(b.logits), b.ids.as<int32_t>(), f(b.weights), stream_);
    const size_t gu = kernels::native_mmvq_weight_bytes(12, 2048, 512);
    gather_resident_experts(b.ids.as<int32_t>(),
                            expert_cache_.resident_slots.as<int32_t>() + il * 256,
                            nullptr, expert_cache_.stride, gu,
                            b.expert_addresses.as<unsigned long long>(),
                            b.expert_rows.as<int32_t>(),
                            b.down_addresses.as<unsigned long long>(),
                            b.down_rows.as<int32_t>(), stream_, expert_cache_.data.device_addresses());
    metrics.expert_cache_hits += 8;
    ++metrics.tg_gpu_hit_layers;
    ++metrics.tg_device_layers;
    batch_moe_grouped_compute(il, layer, 1, "", metrics);
}

void Runtime::grouped_moe_token(size_t il, const LayerWeights& layer, Metrics& metrics) {
    std::chrono::steady_clock::time_point moe_start;
    if (decode_phase_ && tg_diag_) moe_start = std::chrono::steady_clock::now();
    auto& b = batch_;
    cuda_check(cudaMemcpyAsync(b.norm.data(), work_.norm.data(), 2048 * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_), "grouped token norm");
    cuda_check(cudaMemcpyAsync(b.mixer.data(), work_.mixer.data(), 2048 * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_), "grouped token residual");
    batch_moe(il, layer, 1, "", metrics);
    cuda_check(cudaMemcpyAsync(work_.hidden.data(), b.hidden.data(), 2048 * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_), "grouped token output");
    if (decode_phase_ && tg_diag_)
        metrics.diag_moe_wall_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - moe_start).count();
}

} // namespace strata::qwen35moe
