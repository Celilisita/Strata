#pragma once
void dense_folded_ring_pair(const void *weights,const void *input,float *gate,float *up,float *activation,
                       int rows,int columns,int tokens,bool diagnostic,cudaStream_t stream){
    if(rows%64||columns%32||tokens%128||tokens<128)
        throw std::invalid_argument("folded Gate/Up requires full64-row128-token tiles");
    const dim3 grid(tokens/128,rows/64);
    if(diagnostic)dense_folded_ring<true><<<grid,256,0,stream>>>(
        static_cast<const half *>(weights),static_cast<const half *>(input),gate,up,activation,rows,columns,tokens,columns);
    else dense_folded_ring<false><<<grid,256,0,stream>>>(
        static_cast<const half *>(weights),static_cast<const half *>(input),gate,up,activation,rows,columns,tokens,columns);
    check_cuda(cudaGetLastError(),"owned folded Gate/Up SwiGLU");
}
void configure_dense_folded_ring(){
    check_cuda(cudaFuncSetAttribute(dense_folded_ring<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"folded shared policy");
    check_cuda(cudaFuncSetAttribute(dense_folded_ring<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"folded diagnostic shared policy");
    int active=0;check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_folded_ring<false>,256,0),"folded residency");
    if(active<2)throw std::invalid_argument("folded normal path requires two resident CTAs");
}
