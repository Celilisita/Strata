#include "strata/qwen35/native/weight_overlap.hpp"
#include "strata/qwen35/native/ffn.hpp"
namespace strata::qwen35::native {
PpWeightOverlap::PpWeightOverlap(size_t bytes,cudaStream_t main):bytes_(bytes),main_(main){
    if(!bytes||!main||main==cudaStreamLegacy||main==cudaStreamPerThread)
        throw std::invalid_argument("weight overlap requires explicit stream and workspace");
    int stream_device=-1;check_cuda(cudaGetDevice(&device_),"weight preparation device");
    check_cuda(cudaStreamGetDevice(main_,&stream_device),"weight preparation stream device");
    if(stream_device!=device_)throw std::invalid_argument("foreign weight preparation stream");
    size_t free=0,total=0;check_cuda(cudaMemGetInfo(&free,&total),"weight preparation reserve");
    if(bytes>free||free-bytes<size_t(4)*1024*1024*1024)throw std::runtime_error("weight preparation lacks reserve");
    try{
        check_cuda(cudaMalloc(&data_,bytes_),"transient decoded Down operand");
        check_cuda(cudaStreamCreateWithFlags(&prepare_,cudaStreamNonBlocking),"weight preparation stream");
        check_cuda(cudaEventCreateWithFlags(&fork_,cudaEventDisableTiming),"weight preparation fork");
        check_cuda(cudaEventCreateWithFlags(&done_,cudaEventDisableTiming),"weight preparation join");
    }catch(...){release();throw;}
}
void PpWeightOverlap::release()noexcept{
    if(main_)cudaStreamSynchronize(main_);
    if(prepare_)cudaStreamSynchronize(prepare_);
    if(done_)cudaEventDestroy(done_);
    if(fork_)cudaEventDestroy(fork_);
    if(prepare_)cudaStreamDestroy(prepare_);
    if(data_)cudaFree(data_);
    main_=prepare_=nullptr;fork_=done_=nullptr;data_=nullptr;
}
PpWeightOverlap::~PpWeightOverlap(){release();}
void PpWeightOverlap::begin(Matrix weights,cudaStream_t stream){
    matrix_bytes(weights);
    if(stream!=main_||2*size_t(weights.rows)*weights.columns>bytes_)
        throw std::invalid_argument("weight preparation lease changed");
    // In captured execution these ordinary events become graph dependencies.
    // No external event node, host callback, allocation or synchronization.
    check_cuda(cudaEventRecord(fork_,main_),"weight preparation reuse edge");
    check_cuda(cudaStreamWaitEvent(prepare_,fork_,0),"weight preparation fork edge");
    decode_linear(weights,data_,prepare_);
    check_cuda(cudaEventRecord(done_,prepare_),"weight preparation completion edge");
}
const void *PpWeightOverlap::ready(cudaStream_t stream){
    if(stream!=main_)throw std::invalid_argument("weight consumption stream changed");
    check_cuda(cudaStreamWaitEvent(main_,done_,0),"weight preparation join edge");return data_;
}
}
