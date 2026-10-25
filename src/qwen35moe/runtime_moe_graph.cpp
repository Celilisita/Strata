#include "strata/qwen35moe/runtime.hpp"
#include "moe_graph_state.hpp"
#include "tg_phase_lease.hpp"
#include "mtp_state.hpp"
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::prepare_moe_graphs() {
    moe_graph_.reset();
    const char* defer = std::getenv("STRATA_QWEN35_TG_PHASE_DEFER_GRAPH");
    if (defer && std::string(defer) != "0" && std::string(defer) != "1")
        throw std::invalid_argument("TG_PHASE_DEFER_GRAPH must be0 or1");
    const bool defer_capture = defer && std::string(defer) == "1";
    const auto validate_defer = [&] {
        if (defer_capture && (!detail::tg_phase_lease_enabled() || !tg_dynamic_ ||
            mtp_enabled_ || session_capacity() != 1 || decode_phase_ ||
            checkpoint_sink_ || mtp_sink_ || !batch_.capacity ||
            batch_.expert_stage.data() || !expert_cache_.slots || expert_cache_.retention.leased()))
            throw std::invalid_argument("TG_PHASE_DEFER_GRAPH requires TG_PHASE_LEASE=1 and a prepared idle single-session dynamic boundary, MTPoff and no sink");
    };
    const char* recurrent = std::getenv("STRATA_QWEN35_TG_RECURRENT_GRAPH");
    if (recurrent && std::string(recurrent) != "0" && std::string(recurrent) != "1")
        throw std::invalid_argument("TG_RECURRENT_GRAPH must be0 or1");
    const bool capture_recurrent = recurrent && std::string(recurrent) == "1";
    if (capture_recurrent && nvfp4_)
        throw std::invalid_argument("full recurrent graph does not admit typed NVFP4 expert scheduling");
    const bool dense_recurrent=gdn_dense_graph_requested();
    if (capture_recurrent && dense_recurrent)
        throw std::invalid_argument("full recurrent and dense recurrent graph schedules are exclusive");
    const char* flag = std::getenv("STRATA_QWEN35_MOE_GRAPH");
    if (flag && std::string(flag) == "0") {
        if (capture_recurrent || dense_recurrent) throw std::invalid_argument("recurrent graph requires enabled MoE graph");
        validate_defer();
        return;
    }
    if (!flag) {
        if (!tg_dynamic_ || session_capacity() != 1 || decode_phase_) {
            if (capture_recurrent) throw std::invalid_argument("recurrent graph requires single-session dynamic boundary");
            validate_defer();
            return;
        }
        flag = "1";
    }
    if (std::string(flag) != "1" && std::string(flag) != "control")
        throw std::invalid_argument("MOE_GRAPH must be0,1,control");
    if (!tg_dynamic_ || session_capacity() != 1 || decode_phase_)
        throw std::invalid_argument("MoE graph requires prepared single-session dynamic epoch boundary");
    if (capture_recurrent && (mtp_enabled_ || checkpoint_sink_ || mtp_sink_ || std::string(flag) != "1"))
        throw std::invalid_argument("recurrent graph requires MTPoff/no capture sink/enabled MoE graph");
    validate_defer();
    // Phase lease will replace these PP views with capacity16 before TG starts.
    // Keep all flag/eligibility checks above; the reset already retired old graphs.
    if (defer_capture && batch_.capacity > 16) return;
    batch_failed_ = true;
    auto graphs = std::make_unique<MoeGraphState>(expert_sources_.size(), capture_recurrent);
    // Captures enqueue no model computation. The mapped miss worker is started
    // later, at the existing decode epoch. Fixed argument buffers are refreshed
    // there; graph identity excludes their contents, never their addresses.
    decode_phase_ = true;
    try {
        for (size_t il = 0; il < graphs->layers.size(); ++il) {
            const auto& layer = il < layers_.size() ? layers_[il] : mtp_->decoder;
            for (int count = 1; count <= (dynamic_pair_args_.bytes() ? 2 : 1); ++count) {
                auto& entry = graphs->layers[il][size_t(count-1)];
                Metrics capture;
                cuda_check(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal),"begin MoE capture");
                try { batch_moe_device_eager(il,layer,capture,count); }
                catch (...) {
                    cudaGraph_t failed = nullptr;
                    cudaStreamEndCapture(stream_,&failed);
                    if (failed) cudaGraphDestroy(failed);
                    throw;
                }
                cuda_check(cudaStreamEndCapture(stream_,&entry.graph),"finish MoE capture");
                size_t nodes = 0;
                cuda_check(cudaGraphGetNodes(entry.graph,nullptr,&nodes),"count MoE graph nodes");
                if (!nodes || nodes > 32) throw std::runtime_error("unbounded MoE graph capture");
                std::vector<cudaGraphNode_t> list(nodes);
                cuda_check(cudaGraphGetNodes(entry.graph,list.data(),&nodes),"inspect MoE graph nodes");
                for (auto node : list) {
                    cudaGraphNodeType type;
                    cuda_check(cudaGraphNodeGetType(node,&type),"inspect MoE graph node type");
                    if (type != cudaGraphNodeTypeKernel)
                        throw std::runtime_error("MoE graph must contain only canonical kernels");
                }
                entry.kernels = nodes; entry.expert_kernels = capture.expert_launches;
                cuda_check(cudaGraphInstantiate(&entry.exec,entry.graph,0),"instantiate MoE graph");
                cuda_check(cudaGraphUpload(entry.exec,stream_),"upload MoE graph schedule");
            }
        }
        if (capture_recurrent) {
            size_t captured = 0; uint64_t kernel_nodes = 0, copy_nodes = 0;
            for (size_t il = 0; il < spec_.layers.size(); ++il) {
                if (!spec_.layers[il].recurrent) continue;
                auto& entry = graphs->recurrent[il]; Metrics capture;
                // No head/history update on host or epoch start. The recurrent
                // canonical layer has no absolute-position dependence. Capture
                // records these operations; it does not execute model kernels.
                cuda_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin recurrent layer capture");
                try { decode_layer(il, spec_.layers[il], layers_[il], 0, capture); }
                catch (...) {
                    cudaGraph_t failed = nullptr; cudaStreamEndCapture(stream_, &failed);
                    if (failed) cudaGraphDestroy(failed);
                    throw;
                }
                cuda_check(cudaStreamEndCapture(stream_, &entry.graph), "finish recurrent layer capture");
                size_t nodes = 0;
                cuda_check(cudaGraphGetNodes(entry.graph, nullptr, &nodes), "count recurrent graph nodes");
                if (!nodes || nodes > 96) throw std::runtime_error("unbounded recurrent layer graph");
                std::vector<cudaGraphNode_t> list(nodes);
                cuda_check(cudaGraphGetNodes(entry.graph, list.data(), &nodes), "inspect recurrent graph nodes");
                for (auto node : list) {
                    cudaGraphNodeType type;
                    cuda_check(cudaGraphNodeGetType(node, &type), "inspect recurrent graph node type");
                    if (type == cudaGraphNodeTypeKernel) { ++entry.kernels; ++kernel_nodes; }
                    else if (type == cudaGraphNodeTypeMemcpy) {
                        cudaMemcpy3DParms copy{};
                        cuda_check(cudaGraphMemcpyNodeGetParams(node, &copy), "inspect recurrent graph copy");
                        if (copy.kind != cudaMemcpyDeviceToDevice)
                            throw std::runtime_error("recurrent graph copy must be device-to-device");
                        ++copy_nodes;
                    } else throw std::runtime_error("recurrent graph must contain only kernels and D2D copies");
                }
                entry.expert_kernels = capture.expert_launches;
                cuda_check(cudaGraphInstantiate(&entry.exec, entry.graph, 0), "instantiate recurrent graph");
                cuda_check(cudaGraphUpload(entry.exec, stream_), "upload recurrent graph schedule");
                ++captured;
            }
            std::fprintf(stderr, "qwen35_tg_recurrent_graph layers=%zu kernels=%llu d2d_nodes=%llu\n",
                captured, static_cast<unsigned long long>(kernel_nodes), static_cast<unsigned long long>(copy_nodes));
            std::fflush(stderr);
        }
        prepare_gdn_dense_graphs(*graphs);
        cuda_check(cudaStreamSynchronize(stream_),"prepare MoE graph schedules ready");
    } catch (...) { decode_phase_ = false; throw; }
    decode_phase_ = false;
    dynamic_layers_ = dynamic_entries_ = 0;
    graphs->enabled = std::string(flag) == "1";
    graphs->recurrent_enabled = capture_recurrent;
    moe_graph_ = std::move(graphs);
    batch_failed_ = false;
}
void Runtime::invalidate_moe_graphs() { moe_graph_.reset(); }
void Runtime::batch_moe_device(size_t il,const LayerWeights& layer,Metrics& metrics,int count) {
    if (count < 1 || count > 2) throw std::invalid_argument("MoE graph row count");
    wait_nvfp4_admission(il); // Before captured top8 reads the device directory.
    if (moe_graph_) {
        const auto& entry = moe_graph_->layers.at(il)[size_t(count-1)];
        if (!entry.exec) throw std::runtime_error("unprepared MoE graph shape");
        if (moe_graph_->enabled) {
            cuda_check(cudaGraphLaunch(entry.exec,stream_),"launch native MoE schedule");
            moe_graph_->executed_kernel_nodes += entry.kernels;
            metrics.expert_launches += entry.expert_kernels;
            ++metrics.moe_graph_launches; ++metrics.moe_host_submissions;
            ++dynamic_layers_; dynamic_entries_ += size_t(count)*spec_.n_expert_used;
            return;
        }
        metrics.moe_host_submissions += entry.kernels;
    }
    batch_moe_device_eager(il,layer,metrics,count);
}
} // namespace strata::qwen35moe
