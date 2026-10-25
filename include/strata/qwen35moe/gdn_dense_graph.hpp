#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
namespace strata::qwen35moe {
// A submission schedule over stable caller-owned buffers/state. Capture never
// executes it. The owner must drain its stream before invalidate/destruction.
class GdnDenseGraph {
public:
    GdnDenseGraph()=default;
    ~GdnDenseGraph(){invalidate();}
    GdnDenseGraph(const GdnDenseGraph&)=delete;
    GdnDenseGraph& operator=(const GdnDenseGraph&)=delete;
    template<class Launch>void capture(cudaStream_t stream,Launch launch){
        if(!stream||graph_||exec_)throw std::invalid_argument("invalid GDN dense capture state");
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        try{launch();}
        catch(...){cudaGraph_t bad=nullptr;cudaStreamEndCapture(stream,&bad);if(bad)cudaGraphDestroy(bad);throw;}
        try{
            ck(cudaStreamEndCapture(stream,&graph_));ck(cudaGraphGetNodes(graph_,nullptr,&nodes_));
            if(!nodes_||nodes_>64)throw std::runtime_error("unbounded GDN dense graph");
            std::vector<cudaGraphNode_t> list(nodes_);size_t n=nodes_;ck(cudaGraphGetNodes(graph_,list.data(),&n));
            for(auto node:list){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));
                if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("GDN dense graph must be kernel-only");}
            ck(cudaGraphInstantiate(&exec_,graph_,0));ck(cudaGraphUpload(exec_,stream));
        }catch(...){invalidate();throw;}
    }
    void launch(cudaStream_t stream)const{
        if(!stream||!exec_)throw std::runtime_error("unprepared GDN dense graph");
        ck(cudaGraphLaunch(exec_,stream));
    }
    bool ready()const{return exec_!=nullptr;}
    size_t nodes()const{return nodes_;}
    void invalidate()noexcept{if(exec_)cudaGraphExecDestroy(exec_);if(graph_)cudaGraphDestroy(graph_);exec_=nullptr;graph_=nullptr;nodes_=0;}
private:
    static void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
    cudaGraph_t graph_=nullptr;cudaGraphExec_t exec_=nullptr;size_t nodes_=0;
};
}
