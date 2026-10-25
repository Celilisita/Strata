#include "strata/qwen35moe/runtime.hpp"
#include "native_tg_timeline.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "sampling_state.hpp"
#include "head_readback.hpp"
#include "mtp_state.hpp"
#include "moe_graph_state.hpp"
#include "strata/qwen35moe/head_pick.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace strata::qwen35moe {

void Runtime::decode_layer(size_t il, const LayerTensors& source, LayerWeights& layer,
                            int position, Metrics& metrics, int rope_offset) {
    if (decode_phase_ && source.recurrent && moe_graph_ && moe_graph_->recurrent_enabled) {
        const auto& entry = moe_graph_->recurrent.at(il);
        if (!entry.exec) throw std::runtime_error("unprepared recurrent decode graph");
        cuda_check(cudaGraphLaunch(entry.exec, stream_), "launch canonical recurrent layer schedule");
        moe_graph_->executed_kernel_nodes += entry.kernels;
        metrics.expert_launches += entry.expert_kernels;
        ++metrics.moe_graph_launches; ++metrics.moe_host_submissions;
        ++dynamic_layers_; dynamic_entries_ += spec_.n_expert_used;
        return;
    }
    native_tg_timeline::dense_begin(il,source.recurrent,stream_);
    if(!source.recurrent||!launch_gdn_dense_graph(il)){
    kernels::native_qsa_rms_norm_weighted(work_.hidden.as<float>(), layer.attn_norm.f32(),
                                           work_.norm.as<float>(), 2048, 1,
                                           float(spec_.rms_epsilon), stream_);
    const bool mtp_probe = mtp_sink_ && il == spec_.n_layers;
    if (mtp_probe) mtp_sink_->stage(3, work_.norm.as<float>(), 2048, stream_);
    if (source.recurrent) gdn_step(layer);
    else {
        attention_step(layer, position, rope_offset);
        if (checkpoint_sink_ && il == 3 && position == 1)
            checkpoint_sink_->attention(position, work_.q.as<float>(), work_.k.as<float>(),
                                        work_.attn.as<float>(), stream_, metrics);
    }
    if (mtp_probe) mtp_sink_->stage(4, work_.mixer.as<float>(), 2048, stream_);
    kernels::add_inplace(work_.mixer.as<float>(), work_.hidden.as<float>(), 2048, stream_);
    kernels::native_qsa_rms_norm_weighted(work_.mixer.as<float>(), layer.post_norm.f32(),
                                           work_.norm.as<float>(), 2048, 1,
                                           float(spec_.rms_epsilon), stream_);
    if (mtp_probe) mtp_sink_->stage(5, work_.norm.as<float>(), 2048, stream_);
    }
    native_tg_timeline::dense_end(il,stream_);
    if(nvfp4_ && !batch_.capacity)prepare_batch(2); // A one-token first request must also use the typed path.
    if (batch_.capacity) grouped_moe_token(il, layer, metrics);
    else moe_step(il, layer, metrics);
    native_tg_timeline::moe_end(il,stream_);
}

int Runtime::run_token(int token, int position, bool need_logits, Metrics& metrics) {
    if (batch_failed_) throw std::runtime_error("batch workspace failed; recreate runtime");
    if(nvfp4_ && nvfp4_->device_tg && decode_phase_ && !need_logits)
        throw std::invalid_argument("NVFP4 device TG requires the normal token head boundary");
    std::chrono::steady_clock::time_point token_start;
    if (decode_phase_ && tg_diag_) token_start = std::chrono::steady_clock::now();
    if (position < 0 || position >= kv_capacity_ || token < 0 || token >= int(spec_.n_vocab))
        throw std::out_of_range("Qwen35 token or context position");
    const auto timeline_scope=native_tg_timeline::begin(
        nvfp4_&&nvfp4_->device_tg&&tg_dynamic_&&decode_phase_,
        stream_,decode_epoch_,nvfp4_?nvfp4_->tg_token+1:0,spec_.layers.size());
    sampling_history(&token, 1);
    if (mtp_enabled_ && !decode_phase_ && position > 0) {
        mtp_kv_batch(&token, 1, mtp_->seed.as<float>(), position - 1);
        ++metrics.kv_catchup_rows;
    }
    load_embedding(token, position, work_.hidden.as<float>(), spec_.embedding, work_.emb_row, embedding_row_bytes_);

    for (size_t il = 0; il < spec_.layers.size(); ++il) {
        auto& layer = layers_[il];
        submit_nvfp4_admission(il,metrics);
        decode_layer(il, spec_.layers[il], layer, position, metrics);
        // Current Dense/graph is already queued before a later pageable source
        // may block the host. This bounded lookahead introduces no new executor.
        if(il+1<spec_.layers.size())submit_nvfp4_admission(il+1,metrics);
        if (checkpoint_sink_)
            checkpoint_sink_->layer(position, il, work_.hidden.as<float>(), stream_, metrics);
    }
    const int result = need_logits ? head_token(metrics) : -1;
    if(nvfp4_ && nvfp4_->device_tg && tg_dynamic_ && decode_phase_) {
        native_tg_timeline::harvest(metrics,[&](size_t il) {
            return nvfp4_->tg_admission &&
                (nvfp4_->tg_admission->dirty_mask()&(uint64_t(1)<<il))
                ? expert_prefetch_->decode_ready(il) : nullptr;
        });
        replay_nvfp4_routes(metrics);
    }
    if (mtp_enabled_ && !decode_phase_) {
        if (need_logits) cuda_check(cudaMemcpyAsync(mtp_->seed.data(), work_.norm.data(), 2048 * sizeof(float),
            cudaMemcpyDeviceToDevice, stream_), "MTP main seed");
        else kernels::native_qsa_rms_norm_weighted(work_.hidden.as<float>(), head_norm_.f32(),
            mtp_->seed.as<float>(), 2048, 1, float(spec_.rms_epsilon), stream_);
    }
    if (decode_phase_ && tg_diag_)
        metrics.diag_token_wall_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - token_start).count();
    if (tg_dynamic_ && decode_phase_) collect_dynamic_metrics(metrics);
    return result;
}

int Runtime::head_token(Metrics& metrics, const GpuWeight* norm, const GpuWeight* output) {
    collect_visual_metrics(metrics);
    native_tg_timeline::head_begin(stream_);
    selected_logits_ = work_.head_logits.as<float>();
    std::chrono::steady_clock::time_point head_start;
    if (decode_phase_ && tg_diag_) head_start = std::chrono::steady_clock::now();
    kernels::native_qsa_rms_norm_weighted(work_.hidden.as<float>(), norm ? norm->f32() : head_norm_.f32(),
                                           work_.norm.as<float>(), 2048, 1,
                                           float(spec_.rms_epsilon), stream_);
    kernels::native_quantize_q8_1(work_.norm.as<float>(), work_.q8_input.data(), 2048, 1, stream_);
    project(output ? *output : head_, work_.q8_input.data(), work_.head_logits.as<float>(), 2048, int(spec_.n_vocab));
    if (sampling_) launch_sample(metrics);
    else if (gpu_head_)
        select_head_greedy(work_.head_logits.as<float>(), int(spec_.n_vocab),
                           head_partials_.as<HeadPick>(), head_pick_buffer_.as<HeadPick>(), stream_,
                           tg_dynamic_ && decode_phase_
                               ? &dynamic_frame_.as<core::NativeMissFrame>()->failed : nullptr);
    native_tg_timeline::head_ready(stream_);
    std::chrono::steady_clock::time_point copy_start;
    copy_start = std::chrono::steady_clock::now();
    if (sampling_) {
        cuda_check(cudaMemcpyAsync(head_readback_ ? head_readback_->data(sizeof(kernels::SamplePick)) : &sampling_->pick,
            sampling_->gpu.result(), sizeof(kernels::SamplePick),
            cudaMemcpyDeviceToHost, stream_), "read sampled Qwen35 pick");
        metrics.head_d2h += sizeof(kernels::SamplePick);
    } else if (gpu_head_) {
        cuda_check(cudaMemcpyAsync(head_readback_ ? head_readback_->data(sizeof(HeadPick)) : &host_pick_,
                                   head_pick_buffer_.data(), sizeof(HeadPick),
                                   cudaMemcpyDeviceToHost, stream_), "read Qwen35 head pick");
        metrics.head_d2h += sizeof(HeadPick);
    } else {
        cuda_check(cudaMemcpyAsync(host_logits_.data(), work_.head_logits.data(),
                                   host_logits_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream_), "read Qwen35 logits");
        metrics.head_d2h += host_logits_.size() * sizeof(float);
    }
    if (decode_phase_ && tg_diag_)
        metrics.diag_head_copy_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copy_start).count();
    const auto wait_start = std::chrono::steady_clock::now();
    const auto waited=cudaStreamSynchronize(stream_);
    if(waited!=cudaSuccess && nvfp4_ && nvfp4_->device_tg)batch_failed_=true;
    cuda_check(waited, "wait for Qwen35 logits");
    metrics.head_wait_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wait_start).count();
    metrics.head_observe_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - copy_start).count();
    ++metrics.explicit_syncs;
    if (head_readback_) {
        if (sampling_) head_readback_->copy_to(&sampling_->pick, sizeof(kernels::SamplePick));
        else if (gpu_head_) head_readback_->copy_to(&host_pick_, sizeof(HeadPick));
    }
    std::chrono::steady_clock::time_point scan_start;
    if (decode_phase_ && tg_diag_) scan_start = std::chrono::steady_clock::now();
    if (sampling_) {
        host_pick_ = {sampling_->pick.id, sampling_->pick.first_bad, sampling_->pick.logit};
        if (sampling_->verify) check_sample_pick();
    }
    int result = -1;
    if (gpu_head_) {
        if (host_pick_.first_bad <= -2)
            throw std::runtime_error("Qwen35 device continuation failed: " +
                                     std::to_string(-2 - host_pick_.first_bad));
        if (host_pick_.first_bad >= 0)
            throw std::runtime_error("nonfinite Qwen35 logit at vocabulary ID " +
                                     std::to_string(host_pick_.first_bad));
        result = host_pick_.id;
        if (result < 0 || result >= int(spec_.n_vocab))
            throw std::runtime_error("invalid Qwen35 GPU head pick");
    } else {
        for (size_t i = 0; i < host_logits_.size(); ++i)
            if (!std::isfinite(host_logits_[i]))
                throw std::runtime_error("nonfinite Qwen35 logit at vocabulary ID " + std::to_string(i));
        result = int(std::max_element(host_logits_.begin(), host_logits_.end()) - host_logits_.begin());
        host_pick_ = {result, -1, host_logits_[size_t(result)]};
    }
    if (decode_phase_ && tg_diag_) {
        const auto done = std::chrono::steady_clock::now();
        metrics.diag_head_scan_ms += std::chrono::duration<double, std::milli>(
            done - scan_start).count();
        metrics.diag_head_wall_ms += std::chrono::duration<double, std::milli>(
            done - head_start).count();
    }
    return result;
}

float Runtime::last_logit(int token) const {
    if (token < 0 || size_t(token) >= host_logits_.size())
        throw std::out_of_range("Qwen35 diagnostic logit ID");
    if (!gpu_head_) return host_logits_[size_t(token)];
    if (token == host_pick_.id) return host_pick_.logit;
    if (!selected_logits_) throw std::runtime_error("no selected Qwen35 logits");
    float value = 0.0f;
    cuda_check(cudaMemcpyAsync(&value, selected_logits_ + token, sizeof(float),
                               cudaMemcpyDeviceToHost, stream_), "read requested Qwen35 logit");
    cuda_check(cudaStreamSynchronize(stream_), "wait requested Qwen35 logit");
    return value;
}

} // namespace strata::qwen35moe
