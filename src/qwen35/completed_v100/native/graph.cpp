#include "strata/qwen35/completed_v100/native/graph.hpp"
#include <vector>

namespace strata::completed_v100::qwen35::native {
CapturedFfn::CapturedFfn(Ffn &ffn, const float *input, int tokens, cudaStream_t stream)
    : stream_(stream) {
    if (!ffn.owned_provider()) throw std::invalid_argument("capture requires an owned compute provider");
    if (!stream_ || stream_ == cudaStreamLegacy || stream_ == cudaStreamPerThread)
        throw std::invalid_argument("capture requires an explicit device-bound stream");
    check_cuda(cudaGetDevice(&device_), "native capture device");
    int stream_device = -1;
    check_cuda(cudaStreamGetDevice(stream_, &stream_device), "native capture stream device");
    if (stream_device != device_) throw std::invalid_argument("capture device and stream mismatch");
    check_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "native capture begin");
    try {
        ffn.execute(input, tokens, stream_);
    } catch (...) {
        cudaStreamEndCapture(stream_, &graph_);
        if (graph_) cudaGraphDestroy(graph_);
        throw;
    }
    try {
        check_cuda(cudaStreamEndCapture(stream_, &graph_), "native capture end");
        check_cuda(cudaGraphGetNodes(graph_, nullptr, &nodes_), "native graph node count");
        if (nodes_ != ffn.launches()) throw std::runtime_error("unexpected owned graph structure");
        std::vector<cudaGraphNode_t> nodes(nodes_);
        check_cuda(cudaGraphGetNodes(graph_, nodes.data(), &nodes_), "native graph nodes");
        for (auto node : nodes) {
            cudaGraphNodeType type{};
            check_cuda(cudaGraphNodeGetType(node, &type), "native graph node type");
            if (type != cudaGraphNodeTypeKernel) throw std::runtime_error("owned FFN graph must contain only kernels");
        }
        check_cuda(cudaGraphInstantiateWithFlags(&executable_, graph_, 0), "native graph instantiate");
    } catch (...) {
        if (executable_) cudaGraphExecDestroy(executable_);
        if (graph_) cudaGraphDestroy(graph_);
        throw;
    }
}
CapturedFfn::~CapturedFfn() {
    if (executable_) cudaGraphExecDestroy(executable_);
    if (graph_) cudaGraphDestroy(graph_);
}
void CapturedFfn::launch() {
    // The explicit stream and graph are immutable and validated at capture.
    // The executor maintains the CUDA device; replay has no identity polling.
    check_cuda(cudaGraphLaunch(executable_, stream_), "native graph replay");
}
}
