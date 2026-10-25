#include "strata/qwen35/completed_v100/native/recurrent_graph.hpp"

namespace strata::completed_v100::qwen35::native {
CapturedRecurrent::CapturedRecurrent(RecurrentLayer &layer, RecurrentInputs input, RecurrentStateOutputs next,
                                     int tokens, cudaStream_t stream) : stream_(stream) {
    if (!stream_ || stream_ == cudaStreamLegacy || stream_ == cudaStreamPerThread)
        throw std::invalid_argument("owned layer capture requires an explicit stream");
    int device = -1, stream_device = -1;
    check_cuda(cudaGetDevice(&device), "owned layer capture device");
    check_cuda(cudaStreamGetDevice(stream_, &stream_device), "owned layer capture stream device");
    if (device != stream_device) throw std::invalid_argument("owned layer capture device mismatch");
    check_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "owned layer capture begin");
    try { layer.execute(input, next, tokens, stream_); }
    catch (...) {
        cudaStreamEndCapture(stream_, &graph_);
        if (graph_) cudaGraphDestroy(graph_);
        throw;
    }
    try {
        check_cuda(cudaStreamEndCapture(stream_, &graph_), "owned layer capture end");
        check_cuda(cudaGraphGetNodes(graph_, nullptr, &nodes_), "owned layer graph count");
        if (nodes_ != layer.launches(tokens)) throw std::runtime_error("owned layer graph structure changed");
        std::vector<cudaGraphNode_t> nodes(nodes_);
        check_cuda(cudaGraphGetNodes(graph_, nodes.data(), &nodes_), "owned layer graph nodes");
        for (auto node : nodes) {
            cudaGraphNodeType type{}; check_cuda(cudaGraphNodeGetType(node, &type), "owned layer graph node kind");
            if (type != cudaGraphNodeTypeKernel) throw std::runtime_error("owned layer graph contains a non-kernel node");
        }
        check_cuda(cudaGraphInstantiateWithFlags(&executable_, graph_, 0), "owned layer graph instantiate");
    } catch (...) {
        if (executable_) cudaGraphExecDestroy(executable_);
        if (graph_) cudaGraphDestroy(graph_);
        throw;
    }
}
CapturedRecurrent::~CapturedRecurrent() {
    if (executable_) cudaGraphExecDestroy(executable_);
    if (graph_) cudaGraphDestroy(graph_);
}
void CapturedRecurrent::launch() { check_cuda(cudaGraphLaunch(executable_, stream_), "owned whole layer graph replay"); }
}
