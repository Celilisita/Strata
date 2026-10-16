#include "mtp_graph.hpp"
#include <vector>
namespace strata::qwen35::native {
MtpGraphOperation::~MtpGraphOperation(){if(exec_)cudaGraphExecDestroy(exec_);if(graph_)cudaGraphDestroy(graph_);}
void MtpGraphOperation::capture(cudaStream_t stream,const std::function<void()> &enqueue,size_t node_budget){
    if(graph_||exec_||node_budget<1||node_budget>262144)throw std::invalid_argument("invalid MTP graph capture budget");
    check_cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"MTP graph capture begin");
    try{enqueue();}
    catch(...){cudaStreamEndCapture(stream,&graph_);throw;}
    check_cuda(cudaStreamEndCapture(stream,&graph_),"MTP graph capture end");
    std::vector<cudaGraph_t> pending{graph_};
    while(!pending.empty()){
        auto current=pending.back();pending.pop_back();size_t count=0;
        check_cuda(cudaGraphGetNodes(current,nullptr,&count),"MTP actual graph nodes");
        if(!count||nodes_+count>node_budget)throw std::runtime_error("unbounded MTP graph");nodes_+=count;
        std::vector<cudaGraphNode_t> nodes(count);check_cuda(cudaGraphGetNodes(current,nodes.data(),&count),"MTP graph node inventory");
        for(auto node:nodes){
            cudaGraphNodeType type;check_cuda(cudaGraphNodeGetType(node,&type),"MTP graph node type");
            if(type==cudaGraphNodeTypeGraph){cudaGraph_t child;check_cuda(cudaGraphChildGraphNodeGetGraph(node,&child),"MTP child graph");pending.push_back(child);}
            else if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemcpy&&type!=cudaGraphNodeTypeMemset&&type!=cudaGraphNodeTypeEmpty)
                throw std::runtime_error("MTP graph contains host or foreign scheduling node");
        }
    }
    check_cuda(cudaGraphInstantiateWithFlags(&exec_,graph_,0),"MTP graph instantiate");
    check_cuda(cudaGraphUpload(exec_,stream),"MTP graph cold upload");
    check_cuda(cudaStreamSynchronize(stream),"MTP graph upload complete");
}
void MtpGraphOperation::launch(cudaStream_t stream){
    if(!exec_)throw std::runtime_error("MTP graph unavailable");
    check_cuda(cudaGraphLaunch(exec_,stream),"MTP actual graph replay");++replays_;
}
}
