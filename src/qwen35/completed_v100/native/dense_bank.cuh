#pragma once
// Same native MMA, K order and output ownership; only the shared vector layout changes.
void dense_bank_pair(const void *weights,const void *input,float *gate,float *up,
                     int rows,int columns,int tokens,cudaStream_t stream){
    launch_joint<true,false,false,true>(weights,input,gate,up,rows,columns,tokens,1,stream);
}
void dense_bank_linear(const void *weights,const void *input,float *output,
                       int rows,int columns,int tokens,cudaStream_t stream){
    launch_joint<false,false,false,true>(weights,input,output,nullptr,rows,columns,tokens,1,stream);
}
void dense_bank_split(const void *weights,const void *input,float *output,float *partial,
                      int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(!partial||splits<2)throw std::invalid_argument("invalid bank-layout partial owner");
    launch_joint<false,false,false,true>(weights,input,partial,nullptr,rows,columns,tokens,splits,stream);
    const size_t count=size_t(rows)*tokens;
    finish_split<<<unsigned((count+255)/256),256,0,stream>>>(partial,output,count,splits);
    check_cuda(cudaGetLastError(),"bank-layout partition reduction");
}
void configure_dense_bank(){
    const auto configure=[](auto kernel){
        check_cuda(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,100),"bank-layout shared policy");
        int active=0;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,256,0),"bank-layout residency");
        if(active<2)throw std::invalid_argument("bank-layout requires two resident CTAs");
    };
    configure(dense_joint<true,false,false,true>);
    configure(dense_joint<false,false,false,true>);
}
