#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_quantize_rows.hpp"
#include <cstdlib>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::qwen35moe {

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
    if (profile_address) metrics.pp_address_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - address_start).count();
    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][3], stream_), "prefill copy end");
    std::chrono::steady_clock::time_point table_copy_start;
    if (measure_table) table_copy_start = std::chrono::steady_clock::now();
    const bool gathered_gu = (!decode_phase_ && count >= 1024 && prefill_config_.grouped_gu_mmq) ||
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
    const int down_type = int(expert_sources_[il].down_exps->type);
    const bool tc_down = (down_type == 12 || down_type == 14) && b.routed_tc && routed_tc_enabled_ &&
        !decode_phase_ && count >= 1024;
    if (!tc_down) {
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

void Runtime::batch_moe_grouped_compute(size_t il, const LayerWeights& layer, int count,
                                        const std::string& reference, Metrics& metrics) {
    auto& b = batch_;
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    const int entries = count * 8;
    const bool dynamic = tg_dynamic_ && decode_phase_ &&
        (count == 1 || (count == 2 && dynamic_pair_args_.bytes()));

    batch_quantize_q8(f(b.norm), count, 2048);
    if (checkpoint_sink_) {
        checkpoint_sink_->gdn_detail("moe_input_q8", b.input_q8.data(),
            kernels::native_q8_1_bytes(2048) * count, true, stream_);
        checkpoint_sink_->gdn_detail("route_ids", b.ids.data(), size_t(count) * 8 * 4, false, stream_);
    }
    const bool tc = b.routed_tc && routed_tc_enabled_ && !decode_phase_ && count >= 1024;
    uint64_t tc_launches = 0;
    uint64_t gu_launches = batch_moe_gu_mmq(count, metrics);
    if (!gu_launches) {
        if (tc && routed_gu_gathered_) {
            gu_launches = 0;
            for (int part = 0; part < 2; ++part) {
                const auto work = b.routed_tc->run(12,
                    b.host_expert_addresses.data() + size_t(part) * entries,
                    b.host_expert_rows.data() + size_t(part) * entries,
                    entries, b.input_q8.data(), f(b.expert_gu) + size_t(part) * entries * 512,
                    2048, 512, true, prefill_config_.q4_gu_reuse ? 2 : 1);
                metrics.control_h2d += work.control_h2d;
                gu_launches += work.launches;
            }
        } else {
            gu_launches = 1;
            // Feature-off and TG preserve the canonical one-column route schedule.
            kernels::native_mmvq_grouped(12, b.expert_addresses.as<unsigned long long>(),
                b.expert_rows.as<int32_t>(), b.input_q8.data(), f(b.expert_gu),
                2048, 512, entries * 2, stream_);
        }
    }
    swiglu(f(b.expert_gu), f(b.expert_gu) + size_t(entries) * 512,
           f(b.expert_hidden), entries * 512, stream_,
           dynamic ? b.down_addresses.as<unsigned long long>() : nullptr, 512);
    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][5], stream_), "prefill GU end");
    const char* setting = std::getenv("STRATA_QWEN35_BULK_QUANT");
    const bool bulk = !decode_phase_ && setting && std::string(setting) == "1";
    const size_t quant_launches = kernels::native_quantize_rows(f(b.expert_hidden),
        b.expert_q8.data(), 512, entries, bulk, stream_,
        dynamic ? b.down_addresses.as<unsigned long long>() : nullptr);
    const int down_type = int(expert_sources_[il].down_exps->type);
    const uint64_t mmq_launches = batch_moe_down_mmq(il, count, metrics);
    if (mmq_launches) tc_launches += mmq_launches;
    else if (tc && (down_type == 12 || down_type == 14)) {
        const auto work = b.routed_tc->run(down_type, b.host_down_addresses.data(),
            b.host_down_rows.data(), entries, b.expert_q8.data(), f(b.expert_parts), 512, 2048);
        metrics.control_h2d += work.control_h2d; tc_launches += work.launches;
    } else {
    kernels::native_mmvq_grouped(down_type, b.down_addresses.as<unsigned long long>(),
                                 b.down_rows.as<int32_t>(), b.expert_q8.data(),
                                 f(b.expert_parts), 512, 2048, entries, stream_);
    }

    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][6], stream_), "prefill Down end");
    batch_project_native(layer.shared_gate, f(b.shared_gate), count, 2048, 512);
    batch_project_native(layer.shared_up, f(b.shared_up), count, 2048, 512);
    swiglu(f(b.shared_gate), f(b.shared_up), f(b.shared_hidden), count * 512, stream_);
    batch_quantize_q8(f(b.shared_hidden), count, 512);
    batch_project_native(layer.shared_down, f(b.shared_output), count, 512, 2048);
    gate_shared_batch(f(b.shared_output), f(b.logits), count, stream_);
    if (dynamic)
        moe_combine_continue(f(b.expert_parts), f(b.weights), f(b.shared_output),
                             f(b.mixer), f(b.hidden),
                             (count == 2 ? dynamic_pair_args_ : dynamic_args_).as<core::NativeMissArgs>() + il, stream_, count);
    else
        moe_combine_batch(f(b.expert_parts), f(b.weights), f(b.shared_output),
                          f(b.mixer), f(b.hidden), count, stream_);
    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][4], stream_), "prefill MoE end");
    metrics.expert_launches += (tc ? 3 + gu_launches + tc_launches : 4 + gu_launches) + quant_launches +
                               (bulk ? 1 : size_t((count + 7) / 8)) +
        (prefill_tc_ && !decode_phase_ && count >= 1024
            ? 18 + 9 * size_t((count + 1023) / 1024 - 1) : size_t((count + 7) / 8) * 3);
    if (!reference.empty()) {
        batch_probe_.shared = batch_stage_error(reference, "ffn_shexp_gated-0",
                                                 f(b.shared_output), 2048);
        batch_probe_.moe_out = batch_stage_error(reference, "l_out-0",
                                                 f(b.hidden), 2048);
    }
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
