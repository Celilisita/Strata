#pragma once
void dense_folded_compact_pair(const void *weights,const void *input,float *gate,float *up,float *activation,
                       int rows,int columns,int tokens,bool diagnostic,cudaStream_t stream){
    if(rows%32||columns%64||tokens%128||tokens<128)
        throw std::invalid_argument("folded Gate/Up requires full32-row128-token tiles");
    const dim3 grid(tokens/128,rows/32);
    if(diagnostic)dense_folded_compact<true><<<grid,256,0,stream>>>(
        static_cast<const half *>(weights),static_cast<const half *>(input),gate,up,activation,rows,columns,tokens,columns);
    else dense_folded_compact<false><<<grid,256,0,stream>>>(
        static_cast<const half *>(weights),static_cast<const half *>(input),gate,up,activation,rows,columns,tokens,columns);
    check_cuda(cudaGetLastError(),"owned folded Gate/Up SwiGLU");
}
void configure_dense_folded_compact(){
    check_cuda(cudaFuncSetAttribute(dense_folded_compact<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"folded shared policy");
    check_cuda(cudaFuncSetAttribute(dense_folded_compact<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"folded diagnostic shared policy");
    int active=0;check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_folded_compact<false>,256,0),"folded residency");
    if(active<3)throw std::invalid_argument("folded normal path requires three resident CTAs");
}
