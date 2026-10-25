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
    bool fused_gu=false;
    if(tg_gu_cluster_eligible(decode_phase_,dynamic,mtp_enabled_,count,
        layer.shared_gate.type,layer.shared_up.type,int(spec_.n_embd),int(spec_.n_ff_shared),int(spec_.n_expert_used))&&
        !checkpoint_sink_&&!mtp_sink_&&tg_gu_cluster_enabled()){
        TgGuClusterView view{{b.expert_addresses.as<unsigned long long>(),b.expert_rows.as<int32_t>(),
            b.down_addresses.as<unsigned long long>(),b.input_q8.data(),layer.shared_gate.ptr(),layer.shared_up.ptr(),
            f(b.expert_hidden),f(b.shared_hidden)},f(b.expert_gu),f(b.shared_gate),f(b.shared_up)};
        fused_gu=tg_gu_cluster_provider(true,view,stream_);
    }
    uint64_t gu_launches = fused_gu ? 1 : batch_moe_gu_mmq(count, metrics);
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
    const char* setting = std::getenv("STRATA_QWEN35_BULK_QUANT");
    const bool bulk = !decode_phase_ && setting && std::string(setting) == "1";
    const int down_type = int(expert_sources_[il].down_exps->type);
    bool paired_down=false;
    size_t quant_launches = 0;
    const uint64_t fused_launches = batch_moe_down_fused(il,count,metrics);
    if (fused_launches) tc_launches += fused_launches;
    else {
    if(!fused_gu)
    swiglu(f(b.expert_gu), f(b.expert_gu) + size_t(entries) * 512,
           f(b.expert_hidden), entries * 512, stream_,
           dynamic ? b.down_addresses.as<unsigned long long>() : nullptr, 512);
    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][5], stream_), "prefill GU end");
    if(tg_quant_down_pair_eligible(fused_gu,decode_phase_,dynamic,mtp_enabled_,count,
        down_type,layer.shared_down.type,512,2048,int(spec_.n_expert_used))&&tg_quant_down_pair_enabled()){
        kernels::NativeTgQuantDownPair view{f(b.expert_hidden),f(b.shared_hidden),
            b.down_addresses.as<unsigned long long>(),b.down_rows.as<int32_t>(),b.expert_q8.data(),
            b.input_q8.data(),layer.shared_down.ptr(),f(b.expert_parts),f(b.shared_output)};
        paired_down=tg_quant_down_pair_provider(true,down_type,view,stream_);
        if(paired_down){b.half_width=b.half_count=0;dense_profile_.launches(2);}
    }
    // Current Mini asset only; direct producer succeeds before native Q8 may be skipped.
    bool direct_down=false;prefill::mmq::NativeGroupedMmqWork direct_work{};
    if(!paired_down && !decode_phase_ && !mtp_enabled_ && !checkpoint_sink_ && !mtp_sink_ &&
       count>=1024 && count<=2048 && prefill_config_.grouped_down_mmq && b.down_mmq &&
       spec_.n_layers==40 && spec_.n_mtp_layers==0 && gguf_.file_size()==19889400736ull &&
       prefill::mmq::down_direct_packed_enabled()) {
        void* packed=b.qkv.data();size_t bytes=b.qkv.bytes();
        if(b.phase_arena.data()) {const size_t offset=size_t(b.capacity)*20480*sizeof(float);
            if(offset>b.phase_arena.bytes())throw std::runtime_error("invalid Down direct arena");
            packed=b.phase_arena.as<uint8_t>()+offset;bytes=b.phase_arena.bytes()-offset;}
        direct_down=b.down_mmq->try_run_down_f32(down_type,b.host_down_addresses.data(),b.host_down_rows.data(),
            entries,entries,f(b.expert_hidden),b.expert_addresses.data(),b.expert_addresses.bytes(),packed,bytes,
            f(b.expert_parts),direct_work);
        if(direct_down){metrics.control_h2d+=direct_work.control_bytes;++b.down_mmq_calls;tc_launches+=direct_work.launches;}
    }
    quant_launches = direct_down?0:paired_down?1:kernels::native_quantize_rows(f(b.expert_hidden),
        b.expert_q8.data(), 512, entries, bulk, stream_,
        dynamic ? b.down_addresses.as<unsigned long long>() : nullptr);
    if(!paired_down && !direct_down){
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

    } // Existing expert Down fallback.
    } // Optional SM120 fused Down.
    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][6], stream_), "prefill Down end");
    static const bool shared_raw=shared_f16_gemm_policy(std::getenv("STRATA_QWEN35_SHARED_F16_GEMM"));
    const bool shared_f16=shared_raw&&!decode_phase_&&prefill_tc_&&count>=1024&&count<=4096&&
        !checkpoint_sink_&&!paired_session_&&!sessions_&&!mtp_enabled_&&!verify_state_&&!verify_history_&&
        layer.shared_gate.type==12&&layer.shared_up.type==12&&
        (layer.shared_down.type==12||layer.shared_down.type==13);
    if(shared_f16) {
        if(dense_profile_.active())throw std::invalid_argument("shared F16 GEMM internal cuBLAS launch profile unknown");
        if(b.gemm.stream()!=stream_)throw std::invalid_argument("shared F16 GEMM stream identity");
        b.half_width=0;b.half_count=0;b.clear_raw_input();
        run_shared_f16_gemm(b.gemm,{layer.shared_gate.ptr(),layer.shared_gate.data.bytes(),layer.shared_gate.type},
            {layer.shared_up.ptr(),layer.shared_up.data.bytes(),layer.shared_up.type},
            {layer.shared_down.ptr(),layer.shared_down.data.bytes(),layer.shared_down.type},
            f(b.norm),b.input_half.as<uint16_t>(),b.input_half.bytes(),f(b.shared_gate),f(b.shared_up),
            f(b.shared_hidden),f(b.shared_output),f(b.logits),count);
    } else {
    if(!fused_gu){
        batch_project_native(layer.shared_gate, f(b.shared_gate), count, 2048, 512);
        batch_project_native(layer.shared_up, f(b.shared_up), count, 2048, 512);
        swiglu(f(b.shared_gate), f(b.shared_up), f(b.shared_hidden), count * 512, stream_);
    }
    if(!paired_down){
    batch_quantize_q8(f(b.shared_hidden), count, 512);
    batch_project_native(layer.shared_down, f(b.shared_output), count, 512, 2048);
    }
    gate_shared_batch(f(b.shared_output), f(b.logits), count, stream_);
    } // shared F16 / original shared branch
    if (dynamic)
        moe_combine_continue(f(b.expert_parts), f(b.weights), f(b.shared_output),
                             f(b.mixer), f(b.hidden),
                             (count == 2 ? dynamic_pair_args_ : dynamic_args_).as<core::NativeMissArgs>() + il, stream_, count);
    else
        moe_combine_batch(f(b.expert_parts), f(b.weights), f(b.shared_output),
                          f(b.mixer), f(b.hidden), count, stream_);
    if (!prefill_events_.empty() && !decode_phase_)
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][4], stream_), "prefill MoE end");
    metrics.expert_launches += (tc ? 3 + gu_launches + tc_launches : 4 + gu_launches) -
                               (fused_launches ? 1 : 0) + quant_launches +
        // Legacy submission estimate: vendor API calls are not private device-kernel counts.
        (shared_f16 ? 10 : (bulk ? 1 : size_t((count + 7) / 8)) +
            (prefill_tc_ && !decode_phase_ && count >= 1024
                ? 18 + 9 * size_t((count + 1023) / 1024 - 1) : size_t((count + 7) / 8) * 3));
    if(fused_gu) metrics.expert_launches-=4; // Four actual expert kernels were deleted, not host graph replays.
    if(paired_down) metrics.expert_launches-=2; // Two actual quant/Down kernels, not graph replay submissions.
    if (!reference.empty()) {
        batch_probe_.shared = batch_stage_error(reference, "ffn_shexp_gated-0",
                                                 f(b.shared_output), 2048);
        batch_probe_.moe_out = batch_stage_error(reference, "l_out-0",
                                                 f(b.hidden), 2048);
    }
}


} // namespace strata::qwen35moe
