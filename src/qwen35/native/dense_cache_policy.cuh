#pragma once
// Cold driver hint. Integer truncation avoids selecting the next partition
// when two CTAs fit an exact supported boundary. Runtime cost admits policy.
template<class Kernel>
void configure_dense_cache(Kernel kernel){
    int device=0;cudaDeviceProp caps{};cudaFuncAttributes function{};
    check_cuda(cudaGetDevice(&device),"shared cache device");
    check_cuda(cudaGetDeviceProperties(&caps,device),"shared cache capabilities");
    check_cuda(cudaFuncGetAttributes(&function,kernel),"shared cache kernel resources");
    const size_t per_block=function.sharedSizeBytes+caps.reservedSharedMemPerBlock;
    const size_t requested=2*per_block,available=caps.sharedMemPerMultiprocessor;
    int hint=available&&requested<available?int(100*requested/available):100;
    if(hint<1)hint=1;
    check_cuda(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,hint),"shared cache carve hint");
    int predicted=0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&predicted,kernel,256,0),"shared cache occupancy prediction");
    if(predicted<2){
        hint=100;
        check_cuda(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,hint),"shared cache fallback hint");
    }
}
