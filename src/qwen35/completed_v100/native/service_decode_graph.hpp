#pragma once
#include "strata/qwen35/completed_v100/native/model.hpp"
#include <array>

namespace strata::completed_v100::qwen35::native {
// A request owns these two parity graphs. KV is reserved before construction.
// The one-token boundary lets the real sampled ID feed the next graph replay.
class ServiceDecodeGraph {
    DenseModel& model_;
    cudaStream_t stream_;
    DeviceStorage storage_;
    DecodeCapture capture_{};
    std::array<cudaGraph_t, 2> graphs_{};
    std::array<cudaGraphExec_t, 2> execs_{};
    int maximum_;
    unsigned parity() const {
        unsigned value = 2;
        for (const auto& state : model_.states_) if (state.state) {
            if (value == 2) value = state.current;
            else if (value != state.current) throw std::logic_error("Dense GDN parity disagrees");
        }
        if (value > 1) throw std::logic_error("Dense GDN state is absent");
        return value;
    }
    void parity(unsigned value) {
        for (auto& state : model_.states_) if (state.state) state.current = value;
    }
    void release() noexcept {
        cudaStreamSynchronize(stream_);
        for (auto exec : execs_) if (exec) cudaGraphExecDestroy(exec);
        for (auto graph : graphs_) if (graph) cudaGraphDestroy(graph);
    }
public:
    ServiceDecodeGraph(DenseModel& model, int first, int maximum, cudaStream_t stream)
        : model_(model), stream_(stream), maximum_(maximum) {
        model_.validate_captured_decode_pair();
        if (maximum > model_.cache_logical_capacity()) throw std::logic_error("unreserved Dense graph KV");
        capture_.cursor = static_cast<DecodeCursor*>(storage_.allocate(sizeof(DecodeCursor)));
        capture_.ids = static_cast<int*>(storage_.allocate(12));
        capture_.flags = static_cast<unsigned*>(storage_.allocate(12));
        capture_.capacity = 3; capture_.partitions = (maximum + 255) / 256;
        const unsigned original = parity();
        try {
            for (unsigned p = 0; p != 2; ++p) {
                parity(p);
                check_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "Dense step capture");
                model_.decode_capture(&capture_);
                try { model_.decode(first, stream_); }
                catch (...) { model_.decode_capture(nullptr); cudaStreamEndCapture(stream_, &graphs_[p]); throw; }
                model_.decode_capture(nullptr);
                check_cuda(cudaStreamEndCapture(stream_, &graphs_[p]), "Dense step capture end");
                check_cuda(cudaGraphInstantiateWithFlags(&execs_[p], graphs_[p], 0), "Dense step instantiate");
                check_cuda(cudaGraphUpload(execs_[p], stream_), "Dense step upload");
            }
            parity(original);
        } catch (...) { parity(original); release(); throw; }
    }
    ~ServiceDecodeGraph() { release(); }
    ServiceDecodeGraph(const ServiceDecodeGraph&) = delete;
#ifdef STRATA_NATIVE_STATE_CHECK
    struct CheckNodes {size_t total=0,kernels=0,memsets=0,copies=0,other=0;};
    CheckNodes check_nodes() const {
        CheckNodes result;size_t count=0;
        check_cuda(cudaGraphGetNodes(graphs_[0],nullptr,&count),"State probe graph cardinality");
        std::vector<cudaGraphNode_t> nodes(count);
        check_cuda(cudaGraphGetNodes(graphs_[0],nodes.data(),&count),"State probe graph nodes");
        result.total=count;
        for(auto node:nodes){cudaGraphNodeType type;
            check_cuda(cudaGraphNodeGetType(node,&type),"State probe graph node type");
            if(type==cudaGraphNodeTypeKernel)++result.kernels;
            else if(type==cudaGraphNodeTypeMemset)++result.memsets;
            else if(type==cudaGraphNodeTypeMemcpy)++result.copies;
            else ++result.other;
        }
        return result;
    }
    // Correctness-only: submit the exact captured cursor policy eagerly.
    void check_eager(int position) {
        initialize_decode_cursor(capture_.cursor,position,stream_);
        model_.decode_capture(&capture_);
        try { model_.decode(position,stream_); }
        catch (...) { model_.decode_capture(nullptr);throw; }
        model_.decode_capture(nullptr);
    }
#endif
    void launch(int position) {
        if (position < 1 || position >= maximum_) throw std::out_of_range("Dense graph request position");
        // The last legal context positions use the ordinary kernel entry, whose
        // cursor initializer deliberately has a stricter bounded graph extent.
        if (position > 262142) { model_.decode(position, stream_); return; }
        const unsigned p = parity();
        initialize_decode_cursor(capture_.cursor, position, stream_);
        check_cuda(cudaGraphLaunch(execs_[p], stream_), "Dense sampled step replay");
        parity(p ^ 1);
    }
};
} // namespace strata::completed_v100::qwen35::native
