#include "pp_prefetch_core.hpp"
#include "nvfp4_decode8.hpp"
#include "p2_vector_encode.hpp"
#include "strata/qwen35/completed_v100/native/gdn_subwarp16.hpp"
#include <algorithm>
namespace strata::completed_v100::qwen35::native {
namespace {
void checked(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("PP prefetch cuBLAS "+std::to_string(int(status)));}
}
PrefetchCore::PrefetchCore(Nvfp4BlasLease b):handle(b.handle),vendor(b.workspace),vendor_bytes(b.bytes),one(b.one),zero(b.zero){
    if(!handle||!vendor||vendor_bytes<32*1024*1024||!one||!zero)throw std::invalid_argument("PP prefetch missing parent cuBLAS lease");
    try{
        check_cuda(cudaStreamCreateWithFlags(&auxiliary,cudaStreamNonBlocking),"PP prefetch auxiliary");
        check_cuda(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming),"PP prefetch fork");
        check_cuda(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),"PP prefetch ready");
    }catch(...){if(fork)cudaEventDestroy(fork);if(ready)cudaEventDestroy(ready);if(auxiliary)cudaStreamDestroy(auxiliary);throw;}
}
PrefetchCore::~PrefetchCore(){
    if(main)cudaStreamSynchronize(main);if(auxiliary)cudaStreamSynchronize(auxiliary);entries.clear();
    if(fork)cudaEventDestroy(fork);if(ready)cudaEventDestroy(ready);if(auxiliary)cudaStreamDestroy(auxiliary);
    // Parent projection owns handle/workspace/scalars and is destroyed after this core.
}
void PrefetchCore::bind(cudaStream_t stream){
    if(main&&main!=stream)throw std::invalid_argument("PP prefetch stream owner changed without reset");
    if(!main)main=stream;
}
void PrefetchCore::reset(cudaStream_t stream){
    if(main&&stream!=main)throw std::invalid_argument("PP prefetch reset on foreign stream");
    if(main)check_cuda(cudaStreamSynchronize(main),"PP prefetch reset drain");
    check_cuda(cudaStreamSynchronize(auxiliary),"PP prefetch auxiliary drain");
    entries.clear();lease={};pending=enabled=false;calls=built=reused=0;main=nullptr;
}
PrefetchCore::Key PrefetchCore::key()const{
    const auto& l=lease;const auto& x=l.input;const auto& s=l.shape;
    return {uintptr_t(x.q),uintptr_t(x.k),uintptr_t(x.v),uintptr_t(x.log_decay),uintptr_t(x.beta),uintptr_t(x.state),
        uintptr_t(l.attention),uintptr_t(l.state),uintptr_t(l.work),uintptr_t(l.scratch.activation),uintptr_t(l.scratch.half_input),
        uintptr_t(l.scratch.gate),uintptr_t(l.gate.data),uintptr_t(l.up.data),uintptr_t(l.down.data),uintptr_t(main),
        uintptr_t(s.key_dim),uintptr_t(s.value_dim),uintptr_t(s.query_heads),uintptr_t(s.value_heads),uintptr_t(s.tokens),
        uintptr_t(l.gate.rows),uintptr_t(l.gate.columns)};
}
void PrefetchCore::begin(cudaStream_t stream){
    check_cuda(cudaEventRecord(fork,stream),"PP prefetch fork record");
    check_cuda(cudaStreamWaitEvent(auxiliary,fork),"PP prefetch fork dependency");
    const auto& l=lease;
    nvfp4_decode_raw_half8(static_cast<const unsigned char*>(l.gate.data)+16,l.scratch.activation,l.gate.rows,l.gate.columns,auxiliary);
    nvfp4_decode_raw_half8(static_cast<const unsigned char*>(l.up.data)+16,
        reinterpret_cast<unsigned char*>(l.scratch.activation)+size_t(l.gate.rows)*l.gate.columns*2,l.up.rows,l.up.columns,auxiliary);
    check_cuda(cudaEventRecord(ready,auxiliary),"PP prefetch decoded ready");
}
void PrefetchCore::gdn(cudaStream_t stream){
    auto submit=[&]{prefetch_gdn::run(lease.input,lease.shape,lease.attention,lease.state,lease.work,lease.work_bytes,
        stream,callback,this);check_cuda(cudaStreamWaitEvent(stream,ready),"PP prefetch subgraph branch join");};
    cudaStreamCaptureStatus capture;check_cuda(cudaStreamIsCapturing(stream,&capture),"PP prefetch capture scope");
    if(capture==cudaStreamCaptureStatusActive){submit();return;}
    if(capture!=cudaStreamCaptureStatusNone)throw std::runtime_error("PP prefetch invalid capture state");
    const auto id=key();for(auto& e:entries)if(e.key==id){
        check_cuda(cudaGraphLaunch(e.executable,stream),"PP prefetch cached subgraph");++reused;return;}
    if(entries.size()>=192)throw std::runtime_error("PP prefetch bounded graph directory exceeded");
    Entry entry;entry.key=id;
    check_cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"PP prefetch graph begin");
    try{submit();check_cuda(cudaStreamEndCapture(stream,&entry.graph),"PP prefetch graph end");}
    catch(...){cudaGraph_t discarded=nullptr;cudaStreamEndCapture(stream,&discarded);if(discarded)cudaGraphDestroy(discarded);throw;}
    check_cuda(cudaGraphInstantiateWithFlags(&entry.executable,entry.graph,0),"PP prefetch instantiate");
    check_cuda(cudaGraphLaunch(entry.executable,stream),"PP prefetch cold subgraph");entries.push_back(std::move(entry));++built;
}
void PrefetchCore::ffn(const float* input,float* packed,float* output,int tokens,cudaStream_t stream){
    const auto& l=lease;const int rows=l.gate.rows,columns=l.gate.columns;auto* weights=l.scratch.activation;auto* encoded=l.scratch.half_input;
    checked(cublasSetStream(handle,stream));checked(cublasSetWorkspace(handle,vendor,vendor_bytes));
    checked(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE));
    q8p2_encode_vector(input,encoded,size_t(tokens)*columns,stream);
    checked(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,2*rows,tokens,columns,one,weights,CUDA_R_16F,columns,
        encoded,CUDA_R_16F,columns,zero,packed,CUDA_R_32F,2*rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    swiglu_encode_vector(packed,static_cast<const float*>(l.gate.data),static_cast<const float*>(l.up.data),encoded,rows,tokens,stream);
    nvfp4_decode_raw_half8(static_cast<const unsigned char*>(l.down.data)+16,weights,columns,rows,stream);
    checked(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,columns,tokens,rows,l.down.data,weights,CUDA_R_16F,rows,
        encoded,CUDA_R_16F,rows,zero,output,CUDA_R_32F,columns,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
}
