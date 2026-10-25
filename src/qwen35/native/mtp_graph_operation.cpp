#include "mtp_graph.hpp"
#include <vector>
#include <chrono>
#include <cstdlib>
#include <string>
namespace strata::qwen35::native {
MtpExecStats &mtp_exec_stats(){static thread_local MtpExecStats value;return value;}
MtpRewindDiagnostic &mtp_rewind_diagnostic(){static thread_local MtpRewindDiagnostic value;return value;}
bool defer_mtp_exec(bool chain,bool retained,int capacity){
    const char *p=std::getenv("STRATA_DENSE_MTP_LAZY_EXEC");
    if(p&&std::string(p)!="0"&&std::string(p)!="1")throw std::invalid_argument("MTP exec policy must be0or1");
    if((p&&p[0]=='0')||!chain||!retained||capacity!=2)return false;
    const auto caps=capabilities(0);return caps.major==12&&caps.minor==0;
}
MtpGraphOperation::~MtpGraphOperation(){if(exec_)cudaGraphExecDestroy(exec_);if(graph_)cudaGraphDestroy(graph_);}
void MtpGraphOperation::capture(cudaStream_t stream,const std::function<void()> &enqueue,size_t node_budget,bool eager){
    if(graph_||exec_||node_budget<1||node_budget>262144)throw std::invalid_argument("invalid MTP graph capture budget");
    const auto started=std::chrono::steady_clock::now();stream_=stream;
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
    auto &stats=mtp_exec_stats();++stats.recorded;
    stats.record_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    if(eager)instantiate(stream,false);else ++stats.deferred;
}
void MtpGraphOperation::instantiate(cudaStream_t stream,bool lazy){
    if(!graph_||exec_||stream!=stream_)throw std::logic_error("MTP executable plan lease");
    const auto started=std::chrono::steady_clock::now();
    check_cuda(cudaGraphInstantiateWithFlags(&exec_,graph_,0),"MTP graph instantiate");
    check_cuda(cudaGraphUpload(exec_,stream),"MTP graph cold upload");
    check_cuda(cudaStreamSynchronize(stream),"MTP graph upload complete");
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    auto &stats=mtp_exec_stats();++stats.instantiated;++stats.upload_syncs;stats.instantiate_ms+=ms;
    if(lazy){++stats.lazy_instantiated;stats.lazy_ms+=ms;}
}
void MtpGraphOperation::launch(cudaStream_t stream){
    if(!graph_||stream!=stream_)throw std::runtime_error("MTP graph unavailable or stream lease changed");
    if(!exec_)instantiate(stream,true);
    check_cuda(cudaGraphLaunch(exec_,stream),"MTP actual graph replay");++replays_;
}
}
