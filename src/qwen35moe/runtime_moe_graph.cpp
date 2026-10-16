#include "strata/qwen35moe/runtime.hpp"
#include "moe_graph_state.hpp"
#include "mtp_state.hpp"
#include <cstdlib>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::prepare_moe_graphs() {
    moe_graph_.reset();
    const char* flag = std::getenv("STRATA_QWEN35_MOE_GRAPH");
    if (flag && std::string(flag) == "0") return;
    if (!flag) {
        if (!tg_dynamic_ || session_capacity() != 1 || decode_phase_) return;
        flag = "1";
    }
    if (std::string(flag) != "1" && std::string(flag) != "control")
        throw std::invalid_argument("MOE_GRAPH must be0,1,control");
    if (!tg_dynamic_ || session_capacity() != 1 || decode_phase_)
        throw std::invalid_argument("MoE graph requires prepared single-session dynamic epoch boundary");
    batch_failed_ = true;
    auto graphs = std::make_unique<MoeGraphState>(expert_sources_.size());
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
        cuda_check(cudaStreamSynchronize(stream_),"prepare MoE graph schedules ready");
    } catch (...) { decode_phase_ = false; throw; }
    decode_phase_ = false;
    dynamic_layers_ = dynamic_entries_ = 0;
    graphs->enabled = std::string(flag) == "1";
    moe_graph_ = std::move(graphs);
    batch_failed_ = false;
}
void Runtime::invalidate_moe_graphs() { moe_graph_.reset(); }
void Runtime::batch_moe_device(size_t il,const LayerWeights& layer,Metrics& metrics,int count) {
    if (count < 1 || count > 2) throw std::invalid_argument("MoE graph row count");
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
